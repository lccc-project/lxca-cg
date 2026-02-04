use cmli::{archs::x86::X86Register, compiler::CompilerContext};

use crate::callconv::CallConvSpec;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct X86CallingConvention {
    pub reg_params: &'static [X86Register],
    pub vreg_params: &'static [X86Register],
    pub rtd: bool,
    pub abi_mode: X86AbiMode,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum X86AbiMode {
    SysV,
    Microsoft,
}

pub struct ParamState<'a> {
    pub info: &'a CompilerContext,
}
