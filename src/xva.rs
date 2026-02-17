use std::{collections::HashMap, hash::BuildHasher, ops::Index};

use cmli::{
    compiler::Compiler,
    mach::MachineMode,
    xva::{
        self, Linkage, XvaCategory, XvaDest, XvaExpr, XvaFile, XvaFunction, XvaFunctionDef,
        XvaObjectDef, XvaOpcode, XvaOperand, XvaRegister, XvaStatement, XvaType,
    },
};
use indexmap::{IndexMap, IndexSet};
use lccc_siphash::{RawSipHasher, build::RandomState};
use lccc_targets::properties::target::Target;
use lxca::ir::{
    constant::{Constant, ConstantPool},
    decls::{DeclarationBody, FunctionBody},
    expr::{BasicBlock, Expr, FunctionCall, JumpTarget, Statement, Terminator, Value},
    file::File,
    symbol::Symbol,
    types::Signature,
};

use crate::{
    callconv::{CallConv, CallConvInfo, CallConvLocation},
    helpers::FetchIncrement,
    layout::layout_type,
};

pub type IntrinsicError = ();

pub trait XvaCompiler {
    fn call_conv(&self) -> &dyn CallConv;

    fn machine_mode(&self) -> MachineMode;

    fn compiler(&self) -> &dyn Compiler;

    #[allow(unused)]
    fn lower_intrinsic_call<'ir>(
        &self,
        name: &str,
        const_args: &[Value<'ir>],
        args: &[XvaRegister],
        output: &mut Vec<XvaStatement>,
    ) -> Option<Result<(), IntrinsicError>> {
        None
    }
}

struct XvaLowerer<'ir, 'a> {
    target: &'a Target,
    compiler: &'a (dyn XvaCompiler + 'a),
    local_vars: IndexMap<Constant<'ir, Symbol>, IndexMap<Constant<'ir, Symbol>, XvaRegister>>,
    cc: Option<&'a dyn CallConv>,
    xva_function: XvaFunction,
    constants: &'a ConstantPool<'ir>,
    strings: &'a mut DataMap,
    info: Option<CallConvInfo>,
    vreg_num: u32,
    ret_ptr: Option<XvaRegister>,
    name: Constant<'ir, Symbol>,
    cur_label: Option<Constant<'ir, Symbol>>,
}

impl<'ir, 'a> XvaLowerer<'ir, 'a> {
    fn new(
        compiler: &'a (dyn XvaCompiler + 'a),
        target: &'a Target,
        constants: &'a ConstantPool<'ir>,
        name: Constant<'ir, Symbol>,
        strings: &'a mut DataMap,
    ) -> Self {
        Self {
            compiler,
            target,
            local_vars: IndexMap::new(),
            xva_function: XvaFunction {
                params: Vec::new(),
                preserve_regs: Vec::new(),
                return_reg: Vec::new(),
                statement: Vec::new(),
            },
            cc: None,
            constants,
            info: None,
            vreg_num: 0,
            ret_ptr: None,
            name,
            strings,
            cur_label: None,
        }
    }

    fn lower_cc(&mut self, sig: &Signature<'ir>, param_names: &[Constant<'ir, Symbol>]) {
        let callconv = self.compiler.call_conv();

        self.cc = Some(callconv);

        let info = callconv
            .compute_call_conv(None, sig, self.constants, self.target)
            .unwrap();

        self.xva_function
            .preserve_regs
            .extend_from_slice(&info.preserved_registers);

        let tys = sig.params(self.constants);

        let mode = self.compiler.machine_mode();

        let compiler = self.compiler.compiler();
        let mach = compiler.machine();

        for (n, param) in info.param_map.iter().enumerate() {
            let is_param = info.lxca_param_range.contains(&n);
            let lxca_param = n.saturating_sub(info.lxca_param_range.start);

            let layout =
                is_param.then(|| layout_type(&tys[lxca_param], self.constants, self.target));
            let size = layout.as_ref().map(|v| v.size_align.size);
            let align = layout.as_ref().map(|v| v.size_align.align.get());
            let category = layout.as_ref().map(|v| v.category);

            let reg = match param {
                crate::callconv::CallConvLocation::Void => {
                    let size = size.unwrap_or(0);
                    let align = align.unwrap_or(1);

                    let reg = XvaDest {
                        id: self.vreg_num.fetch_inc(),
                        ty: XvaType {
                            size,
                            align,
                            category: XvaCategory::Null,
                        },
                    };

                    let reg = XvaRegister::Virtual(reg);

                    reg
                }
                crate::callconv::CallConvLocation::Registers(xva_regs) => {
                    self.xva_function.params.extend_from_slice(xva_regs);

                    let size =
                        size.unwrap_or_else(|| xva_regs.iter().map(|v| v.size(mach, mode)).sum());
                    let align = align.unwrap_or_else(|| {
                        xva_regs
                            .iter()
                            .map(|v| v.align(mach, mode))
                            .fold(1, u64::max)
                    });

                    let category = category
                        .or_else(|| xva_regs.iter().map(|v| v.category(mach, mode)).last())
                        .unwrap_or(XvaCategory::Null);

                    let reg = XvaDest {
                        id: self.vreg_num.fetch_inc(),
                        ty: XvaType {
                            size,
                            align,
                            category,
                        },
                    };

                    let reg = XvaRegister::Virtual(reg);

                    reg
                }
                crate::callconv::CallConvLocation::Stack(range) => todo!("stack {range:?}"),
                crate::callconv::CallConvLocation::Memory(xva_register) => {
                    todo!("memory")
                }
                crate::callconv::CallConvLocation::StackMemory(range) => {
                    todo!("stack memory {range:?}")
                }
            };

            if is_param {
                let (_, val) = self.local_vars.get_index_mut(0).unwrap();
                val.insert(param_names[lxca_param], reg);
            }
        }

        match &info.return_loc {
            crate::callconv::CallConvLocation::Void => {}
            crate::callconv::CallConvLocation::Registers(regs) => {
                self.xva_function.return_reg.extend_from_slice(regs);
            }
            crate::callconv::CallConvLocation::Stack(_) => {
                panic!("Cannot return directly on the stack")
            }
            crate::callconv::CallConvLocation::Memory(reg) => {
                let ret_ptr: XvaRegister = XvaRegister::Virtual(XvaDest {
                    id: self.vreg_num.fetch_inc(),
                    ty: reg.ty(mach, mode),
                });
                self.ret_ptr = Some(ret_ptr);

                self.xva_function
                    .statement
                    .push(XvaStatement::Expr(XvaExpr {
                        dest: ret_ptr,
                        dest2: None,
                        op: XvaOpcode::Move(*reg),
                    }));

                if let Some(v) = info.return_place {
                    self.xva_function.return_reg.push(v);
                }
            }
            crate::callconv::CallConvLocation::StackMemory(_) => {
                panic!("Cannot return directly on the stack")
            }
        }

        self.info = Some(info);
    }

    fn lower_expr(&mut self, dest: XvaRegister, expr: &Expr<'ir>) {
        let expr = match expr.body(self.constants) {
            lxca::ir::expr::ExprBody::Interned(constant) => unreachable!(),
            lxca::ir::expr::ExprBody::Const(value) => match value.body(self.constants) {
                lxca::ir::expr::ValueBody::Interned(constant) => unreachable!(),
                lxca::ir::expr::ValueBody::Integer(val) => {
                    let val = val.read(self.constants);
                    if val == 0 {
                        XvaOpcode::ZeroInit // this is a peephole optimization, but we might as well
                    } else if let Ok(val) = val.try_into() {
                        XvaOpcode::Const(cmli::xva::XvaConst::Bits(val))
                    } else {
                        todo!()
                    }
                }
                lxca::ir::expr::ValueBody::StringLiteral(st) => {
                    let st = st.get(self.constants);
                    let sym = self.strings.insert(st.as_bytes());

                    XvaOpcode::Const(cmli::xva::XvaConst::Global(sym, 0))
                }
                lxca::ir::expr::ValueBody::ByteLiteral(st) => {
                    let st = st.get(self.constants);
                    let sym = self.strings.insert(st);

                    XvaOpcode::Const(cmli::xva::XvaConst::Global(sym, 0))
                }
                lxca::ir::expr::ValueBody::GlobalAddr(constant) => {
                    let sym = constant.get(self.constants).to_string();

                    XvaOpcode::Const(xva::XvaConst::Global(cmli::intern::Symbol::intern(sym), 0))
                }
                lxca::ir::expr::ValueBody::LocalAddr(constant) => {
                    let sym = format!(
                        "{}.{}",
                        self.name.get(self.constants),
                        constant.get(self.constants)
                    );

                    XvaOpcode::Const(xva::XvaConst::Label(cmli::intern::Symbol::intern(sym)))
                }
                lxca::ir::expr::ValueBody::Uninit => XvaOpcode::Uninit,
                lxca::ir::expr::ValueBody::Invalid => {
                    self.xva_function
                        .statement
                        .push(XvaStatement::Trap(cmli::xva::XvaTrap::Unreachable));

                    XvaOpcode::Uninit
                }
                lxca::ir::expr::ValueBody::Null => XvaOpcode::ZeroInit,
                lxca::ir::expr::ValueBody::ZeroInit => XvaOpcode::ZeroInit,
                lxca::ir::expr::ValueBody::Struct(struct_value) => todo!(),
            },
            lxca::ir::expr::ExprBody::UnaryOp(unary_op, overflow_behaviour, box_or_constant) => {
                todo!()
            }
            lxca::ir::expr::ExprBody::BinaryOp(bin) => {
                let ty = expr.ty();
                let layout = layout_type(ty, self.constants, self.target);
                let xva_ty = layout.xva_type();
                let left = bin.2.get(self.constants);
                let right = bin.3.get(self.constants);

                let dest_left = XvaRegister::Virtual(XvaDest {
                    id: self.vreg_num.fetch_inc(),
                    ty: xva_ty,
                });
                let dest_right = XvaRegister::Virtual(XvaDest {
                    id: self.vreg_num.fetch_inc(),
                    ty: xva_ty,
                });

                self.lower_expr(dest_left, left);
                self.lower_expr(dest_right, right);

                match bin.0 {
                    lxca::ir::expr::BinaryOp::Add => XvaOpcode::BinaryOp {
                        op: xva::BinaryOp::Add,
                        left: dest_left,
                        right: xva::XvaOperand::Register(dest_right),
                    },
                    lxca::ir::expr::BinaryOp::Sub => XvaOpcode::BinaryOp {
                        op: xva::BinaryOp::Add,
                        left: dest_left,
                        right: xva::XvaOperand::Register(dest_right),
                    },
                    lxca::ir::expr::BinaryOp::Mul => todo!(),
                    lxca::ir::expr::BinaryOp::Div => todo!(),
                    lxca::ir::expr::BinaryOp::Mod => todo!(),
                }
            }
            lxca::ir::expr::ExprBody::ReadField(box_or_constant, constant, constant1) => todo!(),
            lxca::ir::expr::ExprBody::ProjectField(box_or_constant, constant, constant1) => todo!(),
            lxca::ir::expr::ExprBody::Struct(constant, items) => todo!(),
        };

        self.xva_function
            .statement
            .push(XvaStatement::Expr(XvaExpr {
                dest,
                dest2: None,
                op: expr,
            }));
    }

    fn lower_expr_as_operand(&mut self, expr: &Expr<'ir>) -> XvaOperand {
        let layout = layout_type(expr.ty(), self.constants, self.target);

        let xva_ty = layout.xva_type();
        match expr.body(self.constants) {
            lxca::ir::expr::ExprBody::Interned(constant) => unreachable!(),
            lxca::ir::expr::ExprBody::Const(value) => match value.body(self.constants) {
                lxca::ir::expr::ValueBody::Interned(constant) => unreachable!(),
                lxca::ir::expr::ValueBody::Integer(val) => {
                    let val = val.read(self.constants);

                    if let Ok(val) = val.try_into() {
                        XvaOperand::Const(cmli::xva::XvaConst::Bits(val))
                    } else {
                        todo!()
                    }
                }
                lxca::ir::expr::ValueBody::Null => XvaOperand::Const(cmli::xva::XvaConst::Bits(0)),
                lxca::ir::expr::ValueBody::GlobalAddr(constant) => {
                    let sym = constant.get(self.constants).to_string();

                    XvaOperand::Const(xva::XvaConst::Global(cmli::intern::Symbol::intern(sym), 0))
                }
                lxca::ir::expr::ValueBody::LocalAddr(constant) => {
                    let sym = format!(
                        "{}.{}",
                        self.name.get(self.constants),
                        constant.get(self.constants)
                    );

                    XvaOperand::Const(xva::XvaConst::Label(cmli::intern::Symbol::intern(sym)))
                }
                _ => {
                    let reg = XvaRegister::Virtual(XvaDest {
                        id: self.vreg_num.fetch_inc(),
                        ty: xva_ty,
                    });
                    self.lower_expr(reg, expr);
                    XvaOperand::Register(reg)
                }
            },
            _ => {
                let reg = XvaRegister::Virtual(XvaDest {
                    id: self.vreg_num.fetch_inc(),
                    ty: xva_ty,
                });
                self.lower_expr(reg, expr);
                XvaOperand::Register(reg)
            }
        }
    }

    fn lower_stmt(&mut self, stmt: &Statement<'ir>) {
        let cur_block = self.cur_label.unwrap();
        match stmt {
            Statement::Assign(stmt) => {
                let ty = stmt.value.ty();

                let layout = layout_type(ty, self.constants, self.target);

                let xva_ty = XvaType {
                    size: layout.size_align.size,
                    align: layout.size_align.align.get(),
                    category: layout.category,
                };

                let dest = XvaRegister::Virtual(XvaDest {
                    id: self.vreg_num.fetch_inc(),
                    ty: xva_ty,
                });

                self.lower_expr(dest, &stmt.value);

                self.local_vars
                    .get_mut(&cur_block)
                    .unwrap()
                    .insert(stmt.id, dest);
            }
        }
    }

    // First return value is list of parameter registers to give to xva
    // Second is the location of the return value in the caller
    // Third is the list of return registers to give to xva
    // Third is the list of volatile/non-preserved registers
    fn prep_function_call(
        &mut self,
        func: &FunctionCall<'ir>,
        tc_ret_place: Option<XvaRegister>,
    ) -> (
        Vec<XvaRegister>,
        CallConvLocation,
        Vec<XvaRegister>,
        Vec<XvaRegister>,
    ) {
        let call_sig = &func.sig;
        let fn_sig = match func.target.ty().body(self.constants) {
            lxca::ir::types::TypeBody::Pointer(pointer_type) => {
                match pointer_type.ty(self.constants).body(self.constants) {
                    lxca::ir::types::TypeBody::Function(signature) => signature,
                    _ => panic!("Attempted to call not a function"),
                }
            }
            lxca::ir::types::TypeBody::Function(signature) => signature,
            _ => panic!("Attempted to call not a function"),
        };

        let cc = self.compiler.call_conv();

        let info = cc
            .compute_call_conv(Some(call_sig), fn_sig, self.constants, self.target)
            .unwrap();

        for (reg, val) in info.extra_sets {
            self.xva_function
                .statement
                .push(XvaStatement::Expr(XvaExpr {
                    dest: reg,
                    dest2: None,
                    op: XvaOpcode::Const(xva::XvaConst::Bits(val)),
                }));
        }

        let mut param_regs = Vec::new();

        for (param_val, dest) in func
            .params
            .iter()
            .zip(&info.param_map[info.lxca_param_range])
        {
            let layout = layout_type(param_val.ty(), self.constants, self.target);
            let xva_type = layout.xva_type();
            match dest {
                CallConvLocation::Void => {
                    let dest = XvaRegister::Virtual(XvaDest {
                        id: self.vreg_num.fetch_inc(),
                        ty: xva_type,
                    });
                    self.lower_expr(dest, param_val);
                }
                CallConvLocation::Registers(regs) => {
                    param_regs.extend_from_slice(regs);
                    match &**regs {
                        [] => {
                            let dest = XvaRegister::Virtual(XvaDest {
                                id: self.vreg_num.fetch_inc(),
                                ty: xva_type,
                            });
                            self.lower_expr(dest, param_val);
                        }
                        [reg] => {
                            self.lower_expr(*reg, param_val);
                        }
                        [..] => todo!("Multiple registers"),
                    }
                }
                CallConvLocation::Stack(range) => todo!("push to stack"),
                CallConvLocation::Memory(xva_register) => todo!("memory"),
                CallConvLocation::StackMemory(range) => todo!("stack memory"),
            }
        }

        let (ret_place, ret_regs) = match info.return_loc {
            CallConvLocation::Void => (CallConvLocation::Void, vec![]),
            CallConvLocation::Memory(reg) => {
                if let Some(tc_ret_place) = tc_ret_place {
                    self.xva_function
                        .statement
                        .push(XvaStatement::Expr(XvaExpr {
                            dest: reg,
                            dest2: None,
                            op: XvaOpcode::Move(tc_ret_place),
                        }));
                    (CallConvLocation::Registers(vec![reg]), vec![])
                } else {
                    todo!("alloca for memory return")
                }
            }
            CallConvLocation::Registers(regs) => (CallConvLocation::Registers(regs.clone()), regs),

            CallConvLocation::StackMemory(_) => todo!("stack memory"),
            CallConvLocation::Stack(_) => panic!("Cannot return directly on the stack"),
        };

        (param_regs, ret_place, ret_regs, info.volatile_registers)
    }

    fn lower_jump(
        &mut self,
        targ: &JumpTarget<'ir>,
        mut ret_place: Option<(CallConvLocation, XvaType)>,
    ) {
        let cur_label = self.cur_label.unwrap();
        for (n, src_var) in targ.args.iter().enumerate() {
            let dest = *self
                .local_vars
                .get(&targ.target)
                .unwrap()
                .get_index(n)
                .unwrap()
                .1;

            let expr = match src_var.get(self.constants).as_str() {
                "#return" => match ret_place.take().unwrap() {
                    (CallConvLocation::Void, _) => XvaOpcode::Uninit,
                    (CallConvLocation::Registers(regs), _xva_ty) => match &*regs {
                        [] => XvaOpcode::Uninit,
                        [reg] => XvaOpcode::Move(*reg),
                        [..] => todo!("Multiple registers"),
                    },
                    (_loc, _xva_ty) => todo!("memory"),
                },
                _ => {
                    let reg = *self
                        .local_vars
                        .get(&cur_label)
                        .unwrap()
                        .get(src_var)
                        .unwrap();

                    XvaOpcode::Move(reg)
                }
            };

            self.xva_function
                .statement
                .push(XvaStatement::Expr(XvaExpr {
                    dest,
                    dest2: None,
                    op: expr,
                }));
        }

        let label = format!(
            "{}.{}",
            self.name.get(self.constants),
            targ.target.get(self.constants)
        );

        self.xva_function
            .statement
            .push(XvaStatement::Jump(cmli::intern::Symbol::intern(label)));
    }

    fn lower_term(&mut self, stmt: &Terminator<'ir>) {
        match &stmt.body {
            lxca::ir::expr::TerminatorBody::Unreachable => self
                .xva_function
                .statement
                .push(XvaStatement::Trap(cmli::xva::XvaTrap::Unreachable)),
            lxca::ir::expr::TerminatorBody::Jump(targ) => {
                self.lower_jump(targ, None);
            }
            lxca::ir::expr::TerminatorBody::Call(call_term) => {
                let dest = self.lower_expr_as_operand(&call_term.func.target);
                let (param_regs, ret_place, ret_regs, volatile_regs) =
                    self.prep_function_call(&call_term.func, None);

                self.xva_function.statement.push(XvaStatement::Call {
                    dest,
                    params: param_regs,
                    ret_val: ret_regs,
                    call_clobber_regs: volatile_regs,
                });

                let ret_layout = layout_type(
                    call_term.func.sig.ret_ty(self.constants),
                    self.constants,
                    self.target,
                );

                let ret_xva_ty = ret_layout.xva_type();

                self.lower_jump(&call_term.next, Some((ret_place, ret_xva_ty)));
            }
            lxca::ir::expr::TerminatorBody::Tailcall(function_call) => todo!(),
            lxca::ir::expr::TerminatorBody::Return(expr) => {
                let info = self.info.as_ref().unwrap();
                let ty = expr.ty();

                let layout = layout_type(ty, self.constants, self.target);

                let xva_ty = XvaType {
                    size: layout.size_align.size,
                    align: layout.size_align.align.get(),
                    category: layout.category,
                };
                if let Some(val) = self.ret_ptr {
                    let dest = XvaRegister::Virtual(XvaDest {
                        id: self.vreg_num.fetch_inc(),
                        ty: xva_ty,
                    });
                    self.lower_expr(dest, expr);
                    self.xva_function.statement.push(XvaStatement::Write(
                        cmli::xva::XvaOperand::Register(val),
                        dest,
                    ));
                } else {
                    match &info.return_loc {
                        crate::callconv::CallConvLocation::Void => {}
                        crate::callconv::CallConvLocation::Registers(regs) => match &**regs {
                            [] => {}
                            [reg] => {
                                self.lower_expr(*reg, expr);
                            }
                            _ => todo!("multiple registers"),
                        },
                        _ => unreachable!("Memory checked"),
                    }
                }

                self.xva_function.statement.push(XvaStatement::Return);
            }
        }
    }

    fn lower_basic_block(&mut self, bb: &BasicBlock<'ir>) {
        let label = format!(
            "{}.{}",
            self.name.get(self.constants),
            bb.label().get(self.constants)
        );

        let label = cmli::intern::Symbol::intern(label);

        self.cur_label = Some(bb.label());

        for stmt in bb.stmts() {
            self.lower_stmt(stmt);
        }

        self.lower_term(bb.term());
    }

    fn lower_function(&mut self, func: &FunctionBody<'ir>) {
        self.lower_cc(func.signature(), func.param_names());

        for (i, bb) in func.body().unwrap().iter().enumerate() {
            let cur_label = bb.label();
            self.local_vars.insert(cur_label, IndexMap::new());
            let local_vars = self.local_vars.get_mut(&cur_label).unwrap();
            if i != 0 {
                for (param, ty) in bb.params() {
                    let layout = layout_type(ty, self.constants, self.target);

                    let xva_ty = layout.xva_type();

                    let reg = XvaRegister::Virtual(XvaDest {
                        id: self.vreg_num.fetch_inc(),
                        ty: xva_ty,
                    });
                    local_vars.insert(*param, reg);
                }
            }
        }

        for bb in func.body().unwrap() {
            self.lower_basic_block(bb);
        }
    }
}

pub struct DataMap(
    IndexMap<Vec<u8>, cmli::intern::Symbol, RandomState<4, 2>>,
    RandomState<4, 2>,
);

impl DataMap {
    pub fn new() -> DataMap {
        let rs = RandomState::new();

        DataMap(IndexMap::with_hasher(rs.clone()), rs)
    }

    pub fn insert<S: AsRef<[u8]> + Into<Vec<u8>>>(&mut self, data: S) -> cmli::intern::Symbol {
        if let Some(sym) = self.0.get(data.as_ref()) {
            *sym
        } else {
            let hash = self.1.hash_one(data.as_ref());

            let i = self.0.len();

            let name = format!(".__X.{i}.h{hash:016X}");

            let sym = cmli::intern::Symbol::intern(name);

            self.0.insert(data.into(), sym);

            sym
        }
    }
}

pub fn lower_lxca<'ir>(file: &File<'ir>, target: &Target, compiler: &dyn XvaCompiler) -> XvaFile {
    let mut xva_file = XvaFile {
        functions: Vec::new(),
        weak_decls: Vec::new(),
        objects: Vec::new(),
    };
    let mut data = DataMap::new();
    for decl in file.decls() {
        let name = decl.name();
        let name_sym = cmli::intern::Symbol::intern(name.get(file.pool()).to_string());
        let linkage = match decl.linkage() {
            lxca::ir::decls::Linkage::External => Linkage::External,
            lxca::ir::decls::Linkage::Internal | lxca::ir::decls::Linkage::Const => {
                Linkage::Internal
            }
            lxca::ir::decls::Linkage::Weak => Linkage::Weak,
        };

        match decl.body() {
            DeclarationBody::Function(func) => {
                if func.is_extern() {
                    if linkage == Linkage::Weak {
                        xva_file.weak_decls.push(name_sym);
                    }
                    continue; // TODO: Emit weak decl
                }

                let mut lowerer = XvaLowerer::new(compiler, target, file.pool(), name, &mut data);
                lowerer.lower_function(func);
                let func = XvaFunctionDef {
                    body: lowerer.xva_function,
                    linkage,
                    label: name_sym,
                    section: xva::XvaSection::Text,
                };

                xva_file.functions.push(func);
            }
            DeclarationBody::Object(_) => todo!("object"),
        }
    }

    for (data, sym) in data.0.into_iter() {
        xva_file.objects.push(XvaObjectDef {
            ty: XvaType {
                size: data.len() as u64,
                align: 0,
                category: XvaCategory::Int,
            },
            body: data,
            relocs: Vec::new(),
            linkage: Linkage::Internal,
            label: sym,
            section: xva::XvaSection::Text,
        });
    }

    xva_file
}
