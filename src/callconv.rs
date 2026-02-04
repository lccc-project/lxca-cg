use cmli::{
    compiler::{Compiler, CompilerContext},
    xva::XvaRegister,
};
use lxca::ir::{
    constant::ConstantPool,
    types::{IntType, Signature, Type},
};

use core::ops::Range;
use std::marker::PhantomData;

#[derive(Clone, Debug, Hash, PartialEq, Eq, Default)]
pub struct CallConvInfo {
    /// Range of "lxca" parameters in param_map
    pub lxca_param_range: Range<usize>,
    /// The list of locations of each real parameter (after expanding varargs)
    pub param_map: Vec<CallConvLocation>,
    /// Values which must be set in non-parameter registers before the call is performed.
    pub extra_sets: Vec<(XvaRegister, u64)>,
    /// The location where the return value is stored.
    pub return_loc: CallConvLocation,
    /// The register to return the return place pointer
    pub return_place: Option<XvaRegister>,
    /// The Set of registers that are volatile, IE. may be used by the callee without being preserved.
    /// The Caller is responsible for saving these registers as it needs the preserve their values accross a call.
    pub volatile_registers: Vec<XvaRegister>,
    /// The set of non-volatile registers, IE. may be used by the callee, but are guaranteed to preserve the value for the caller.
    /// The Calee is responsible for saving these registers as it needs them.
    ///
    /// The set of all other registers (neither in [`CallConvInfo::volatile_registers`] nor [`CallConvInfo::preserved_registers`]) are reserved.
    /// They generally cannot be used, or have external abi stability requirements that apply on descended calls as well.
    pub preserved_registers: Vec<XvaRegister>,

    /// The total size of the stack region to allocate before the call
    pub stack_param_area: u32,
    /// The size of the area cleaned up by the callee
    pub callee_cleanup_size: u32,
    /// The size of the redzone (asynchronous code preserved region)
    pub redzone_size: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Default)]
pub enum CallConvLocation {
    #[default]
    Void,
    Registers(Vec<XvaRegister>),
    Stack(Range<u64>),
    Memory(XvaRegister),
    StackMemory(Range<u64>),
}

pub trait CallConv {
    /// Computes the calling convention. `fn_sig` provides the actual set of parameters
    fn compute_call_conv<'ir>(
        &self,
        call_sig: Option<&Signature<'ir>>,
        fn_sig: &Signature<'ir>,
        pool: &ConstantPool<'ir>,
        info: &CompilerContext,
    ) -> Result<CallConvInfo, CallConvError>;
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum CallConvError {
    InvalidTag(String),
    Other(&'static str),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum ParameterFragmentClass {
    Integer,
    ScalarFloat,
    Vector,
    Extension,
    Memory,
    Other(&'static str),
}

pub trait CallConvSpec {
    type AssignParamsState<'a>: 'a;

    fn from_name(name: &str, ctx: &CompilerContext) -> Option<Self>
    where
        Self: Sized;

    fn make_state(ctx: &CompilerContext) -> Self::AssignParamsState<'_>
    where
        Self: Sized;

    fn classify_int<F: FnMut(ParameterFragmentClass, u32, u32)>(
        &self,
        bits: u16,
        info: &CompilerContext,
        v: F,
    );

    fn stack_order(&self) -> StackOrder;

    fn replace_with_memory_param(
        &self,
        frags: &[ParameterFragmentClass],
    ) -> Option<ParameterFragmentClass>;

    fn replace_with_memory_return(
        &self,
        frags: &[ParameterFragmentClass],
    ) -> Option<ParameterFragmentClass>;

    fn assign_registers_param(
        &self,
        frags: &[ParameterFragmentClass],
        state: &mut Self::AssignParamsState<'_>,
        is_varargs: bool,
    ) -> Option<Vec<XvaRegister>>;

    fn assign_registers_return(&self, frags: &[ParameterFragmentClass]) -> Vec<XvaRegister>;

    /// If the return place parameter is returned for a given set of fragments specify the class of that return value
    fn return_return_place(
        &self,
        frag: &[ParameterFragmentClass],
    ) -> Option<ParameterFragmentClass>;

    fn volatile_registers(&self) -> Vec<XvaRegister>;

    fn non_volatile_registers(&self) -> Vec<XvaRegister>;

    fn add_assigns(
        &self,
        state: &Self::AssignParamsState<'_>,
        is_varargs: bool,
    ) -> Vec<(XvaRegister, u64)>;

    fn shadow_space(&self, state: &Self::AssignParamsState<'_>) -> u32;
    fn redzone(&self, state: &Self::AssignParamsState<'_>) -> u32;

    fn callee_cleanup_size(&self, state: &Self::AssignParamsState<'_>) -> u32;
}

pub struct Spec<C>(pub PhantomData<C>);

impl<C> CallConv for Spec<C>
where
    C: CallConvSpec,
{
    fn compute_call_conv<'ir>(
        &self,
        call_sig: Option<&Signature<'ir>>,
        fn_sig: &Signature<'ir>,
        pool: &ConstantPool<'ir>,
        ctx: &CompilerContext,
    ) -> Result<CallConvInfo, CallConvError> {
        let tag = fn_sig.tag(pool);

        let cc =
            C::from_name(tag, ctx).ok_or_else(|| CallConvError::InvalidTag(tag.to_string()))?;

        let mut state = C::make_state(ctx);

        let real_sig = call_sig.unwrap_or(fn_sig);

        let (ret_frags, _) = classify_ty(real_sig.ret_ty(pool), pool, &cc, ctx);

        let mut params = Vec::new();

        let mut info = CallConvInfo::default();

        if let Some(ret) = cc.replace_with_memory_return(&ret_frags) {
            let ret_place = cc
                .assign_registers_param(core::slice::from_ref(&ret), &mut state, false)
                .unwrap();

            info.lxca_param_range.start += 1;

            info.return_loc = CallConvLocation::Memory(ret_place[0]);
            params.push(CallConvLocation::Registers(ret_place));

            if let Some(v) = cc.return_return_place(&ret_frags) {
                let reg = cc.assign_registers_return(core::slice::from_ref(&v));
                info.return_place = Some(reg[0]);
            }
        } else if ret_frags.len() == 0 {
            info.return_loc = CallConvLocation::Void;
        } else {
            info.return_loc = CallConvLocation::Registers(cc.assign_registers_return(&ret_frags));
        }

        let fn_sig_params = fn_sig.params(pool);

        let is_fn_sig_varargs = fn_sig.varargs(pool);

        let fixed_arity_len = fn_sig_params.len();

        for (i, ty) in real_sig.params(pool).iter().enumerate() {
            let (frags, _) = classify_ty(ty, pool, &cc, ctx);

            let param_is_varargs = is_fn_sig_varargs && (i >= fixed_arity_len);

            let (is_memory, regs) = if let Some(v) = cc.replace_with_memory_param(&frags) {
                let regs = cc.assign_registers_param(
                    core::slice::from_ref(&v),
                    &mut state,
                    param_is_varargs,
                );

                (true, regs)
            } else {
                (
                    false,
                    cc.assign_registers_param(&frags, &mut state, param_is_varargs),
                )
            };

            match regs {
                Some(regs) => {
                    if is_memory {
                        params.push(CallConvLocation::Memory(regs[0]))
                    } else {
                        params.push(CallConvLocation::Registers(regs))
                    }
                }
                None => todo!("Stack Assignment"),
            }
        }

        info.param_map = params;

        info.lxca_param_range.end = info.param_map.len();

        let extra_sets = cc.add_assigns(&state, is_fn_sig_varargs);

        for (reg, _) in &extra_sets {
            info.param_map.push(CallConvLocation::Registers(vec![*reg]));
        }

        info.redzone_size = cc.redzone(&state);

        info.stack_param_area += cc.shadow_space(&state);

        info.callee_cleanup_size = cc.callee_cleanup_size(&state);

        Ok(info)
    }
}

fn classify_ty<'ir>(
    ty: &Type<'ir>,
    pool: &ConstantPool<'ir>,
    spec: &impl CallConvSpec,
    info: &CompilerContext,
) -> (Vec<ParameterFragmentClass>, Vec<(u32, u32)>) {
    let mut frag_array = Vec::new();
    let mut extent_array = Vec::new();
    match ty.body(pool) {
        lxca::ir::types::TypeBody::Interned(constant) => todo!(),
        lxca::ir::types::TypeBody::Named(constant) => todo!("named type"),
        lxca::ir::types::TypeBody::Integer(int_type) => {
            spec.classify_int(int_type.width, info, |frag, base, len| {
                frag_array.push(frag);
                extent_array.push((base, len));
            });
        }
        lxca::ir::types::TypeBody::Char(bits) => {
            spec.classify_int(*bits, info, |frag, base, len| {
                frag_array.push(frag);
                extent_array.push((base, len));
            });
        }
        lxca::ir::types::TypeBody::Pointer(pointer_type) => {
            spec.classify_int(info.properties.ptr_width, info, |frag, base, len| {
                frag_array.push(frag);
                extent_array.push((base, len));
            });
        }
        lxca::ir::types::TypeBody::Function(signature) => panic!("Cannot Classify Function Types"),
        lxca::ir::types::TypeBody::Void => {}
    }

    (frag_array, extent_array)
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum StackOrder {
    LeftToRight,
    RightToLeft,
}
