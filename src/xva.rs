use std::collections::HashMap;

use cmli::{
    compiler::Compiler,
    mach::MachineMode,
    xva::{
        self, Linkage, XvaCategory, XvaDest, XvaExpr, XvaFile, XvaFunction, XvaFunctionDef,
        XvaOpcode, XvaRegister, XvaStatement, XvaType,
    },
};
use lccc_targets::properties::target::Target;
use lxca::ir::{
    constant::{Constant, ConstantPool},
    decls::{DeclarationBody, FunctionBody},
    expr::{BasicBlock, Expr, Statement, Terminator, Value},
    file::File,
    symbol::Symbol,
    types::Signature,
};

use crate::{
    callconv::{CallConv, CallConvInfo},
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
    local_vars: HashMap<Constant<'ir, Symbol>, XvaRegister>,
    cc: Option<&'a dyn CallConv>,
    xva_function: XvaFunction,
    constants: &'a ConstantPool<'ir>,
    info: Option<CallConvInfo>,
    vreg_num: u32,
    ret_ptr: Option<XvaRegister>,
    name: Constant<'ir, Symbol>,
}

impl<'ir, 'a> XvaLowerer<'ir, 'a> {
    fn new(
        compiler: &'a (dyn XvaCompiler + 'a),
        target: &'a Target,
        constants: &'a ConstantPool<'ir>,
        name: Constant<'ir, Symbol>,
    ) -> Self {
        Self {
            compiler,
            target,
            local_vars: HashMap::new(),
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
                self.local_vars.insert(param_names[lxca_param], reg);
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

                    if let Ok(val) = val.try_into() {
                        XvaOpcode::Const(cmli::xva::XvaConst::Bits(val))
                    } else {
                        todo!()
                    }
                }
                lxca::ir::expr::ValueBody::StringLiteral(constant) => todo!(),
                lxca::ir::expr::ValueBody::ByteLiteral(constant) => todo!(),
                lxca::ir::expr::ValueBody::Null => todo!(),
                lxca::ir::expr::ValueBody::GlobalAddr(constant) => todo!(),
                lxca::ir::expr::ValueBody::LocalAddr(constant) => todo!(),
                lxca::ir::expr::ValueBody::Uninit => XvaOpcode::Uninit,
                lxca::ir::expr::ValueBody::Invalid => {
                    self.xva_function
                        .statement
                        .push(XvaStatement::Trap(cmli::xva::XvaTrap::Unreachable));

                    XvaOpcode::Uninit
                }
                lxca::ir::expr::ValueBody::ZeroInit => XvaOpcode::ZeroInit,
                lxca::ir::expr::ValueBody::Struct(struct_value) => todo!(),
            },
            lxca::ir::expr::ExprBody::UnaryOp(unary_op, overflow_behaviour, box_or_constant) => {
                todo!()
            }
            lxca::ir::expr::ExprBody::BinaryOp(bin) => todo!("binary op"),
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

    fn lower_stmt(&mut self, stmt: &Statement<'ir>) {
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

                self.local_vars.insert(stmt.id, dest);
            }
        }
    }

    fn lower_term(&mut self, stmt: &Terminator<'ir>) {
        match &stmt.body {
            lxca::ir::expr::TerminatorBody::Unreachable => self
                .xva_function
                .statement
                .push(XvaStatement::Trap(cmli::xva::XvaTrap::Unreachable)),
            lxca::ir::expr::TerminatorBody::Jump(jump_target) => todo!(),
            lxca::ir::expr::TerminatorBody::Call(call_term) => todo!(),
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

        for stmt in bb.stmts() {
            self.lower_stmt(stmt);
        }

        self.lower_term(bb.term());
    }

    fn lower_function(&mut self, func: &FunctionBody<'ir>) {
        self.lower_cc(func.signature(), func.param_names());

        for bb in func.body().unwrap() {
            for (param, ty) in bb.params() {
                let layout = layout_type(ty, self.constants, self.target);

                let xva_ty = layout.xva_type();

                if !self.local_vars.contains_key(param) {
                    let reg = XvaRegister::Virtual(XvaDest {
                        id: self.vreg_num.fetch_inc(),
                        ty: xva_ty,
                    });

                    self.local_vars.insert(*param, reg);
                }
            }
        }

        for bb in func.body().unwrap() {
            self.lower_basic_block(bb);
        }
    }
}

pub fn lower_lxca<'ir>(file: &File<'ir>, target: &Target, compiler: &dyn XvaCompiler) -> XvaFile {
    let mut xva_file = XvaFile {
        functions: Vec::new(),
        weak_decls: Vec::new(),
    };
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

                let mut lowerer = XvaLowerer::new(compiler, target, file.pool(), name);
                lowerer.lower_function(func);
                let func = XvaFunctionDef {
                    body: lowerer.xva_function,
                    linkage,
                    label: name_sym,
                    section: xva::XvaSection::Global,
                };

                xva_file.functions.push(func);
            }
            DeclarationBody::Object(_) => todo!("object"),
        }
    }

    xva_file
}
