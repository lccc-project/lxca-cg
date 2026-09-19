use std::{marker::PhantomData, slice::Iter};

use cmli::{archs::m65::{IndexReg, W65, W65Mode, W65Register}, mach::{MachineMode, Register}, traits::IdType};
use lccc_targets::properties::target::Target;

use crate::{callconv::{CallConvSpec, ParameterFragmentClass, Spec}, xva::XvaCompiler};

pub struct W65CallConv;


pub struct W65AssignParams<'a>{
    target: &'a Target,
    index_regs: &'static [IndexReg],
    vregs: &'static [u8],
}

impl CallConvSpec for W65CallConv {
    type AssignParamsState<'a> = W65AssignParams<'a>;

    fn from_name(name: &str, ctx: &lccc_targets::properties::target::Target) -> Option<Self>
    where
        Self: Sized {
        match name {
            "C" => Some(Self),
            _ => None,
        }
    }

    fn make_state(ctx: &lccc_targets::properties::target::Target) -> Self::AssignParamsState<'_>
    where
        Self: Sized {
        W65AssignParams{target: ctx, index_regs: const {&[IndexReg::X, IndexReg::Y] } , vregs: const {&[0, 1, 2, 3]}}
    }

    fn classify_int<F: FnMut(crate::callconv::ParameterFragmentClass, u32, u32)>(
        &self,
        mut bits: u16,
        _: &lccc_targets::properties::target::Target,
        mut v: F,
    ) {
        let mut offset = 0;
        while bits > 0 {
            v(ParameterFragmentClass::Integer, offset, bits.clamp_to(..=32) as u32);
            bits = bits.saturating_sub(32);
            offset += 32;
        }
    }

    fn stack_order(&self) -> crate::callconv::StackOrder {
        crate::callconv::StackOrder::RightToLeft
    }

    fn replace_with_memory_param(
        &self,
        frags: &[(crate::callconv::ParameterFragmentClass, u32)],
    ) -> Option<(crate::callconv::ParameterFragmentClass, u32)> {
        if frags.iter().find(|(v, _)| matches!(v, ParameterFragmentClass::Memory)).is_some() {
            Some((ParameterFragmentClass::Integer, 32))
        } else if frags.len() > 2 {
            Some((ParameterFragmentClass::Integer, 32))
        } else {
            None
        }
    }

    fn replace_with_memory_return(
        &self,
        frags: &[(crate::callconv::ParameterFragmentClass, u32)],
    ) -> Option<(crate::callconv::ParameterFragmentClass, u32)> {
        if frags.iter().find(|(v, _)| matches!(v, ParameterFragmentClass::Memory)).is_some() {
            Some((ParameterFragmentClass::Integer, 32))
        } else if frags.len() > 2 {
            Some((ParameterFragmentClass::Integer, 32))
        } else {
            None
        }
    }

    fn assign_registers_param(
        &self,
        frags: &[(crate::callconv::ParameterFragmentClass, u32)],
        state: &mut Self::AssignParamsState<'_>,
        is_varargs: bool,
    ) -> Option<Vec<cmli::mach::Register>> {
        if is_varargs && frags.len() > 0 {
            return None; // Always pass varargs on stack
        }
        match frags {
            [] => Some(Vec::new()),
            [(_, ..8)] if !state.index_regs.is_empty() => {
                let (&reg, next) = state.index_regs.split_first().unwrap();
                state.index_regs = next;

                let reg: W65Register = reg.into_byte_reg();

                Some(vec![Register::new(reg)])
            }
            [(_, ..16)] if !state.index_regs.is_empty() => {
                let (&reg, next) = state.index_regs.split_first().unwrap();
                state.index_regs = next;

                let reg: W65Register = reg.into_word_reg();

                Some(vec![Register::new(reg)])
            }
            frags => {
                let regs = frags.len();

                let (l, r) = state.vregs.split_at_checked(regs)?;

                state.vregs = r;

                Some(l.iter().copied().map(W65Register::R).map(Register::new).collect())
            }
        }
    }

    fn assign_registers_return(&self, frags: &[(crate::callconv::ParameterFragmentClass, u32)]) -> Vec<cmli::mach::Register> {
        match frags {
            [] => Vec::new(),
            [(_, ..8)] => {
                vec![Register::new(W65Register::Ab)]
            }
            [(_, ..16)] => {
                vec![Register::new(W65Register::Aw)]
            }
            frags => {
                let regs = frags.len();

                let l = [0, 1].into_iter().take(regs);
                l.map(W65Register::R).map(Register::new).collect()
            }
        }
    }

    fn return_return_place(
        &self,
        frag: &[(crate::callconv::ParameterFragmentClass, u32)],
    ) -> Option<(crate::callconv::ParameterFragmentClass, u32)> {
        None
    }

    fn volatile_registers(&self) -> Vec<cmli::mach::Register> {
        [W65Register::Ab, W65Register::Aw, W65Register::Xb, W65Register::Xw, W65Register::Yb, W65Register::Yw, W65Register::B].into_iter()
            .chain((0..4).map(W65Register::R))
            .chain((0..8).map(W65Register::Rw))
            .chain([W65Register::R(7), W65Register::Rw(14), W65Register::Rw(15)])
            .map(Register::new).collect()
    }

    fn non_volatile_registers(&self) -> Vec<cmli::mach::Register> {
        (4..7).map(W65Register::R).chain((8..14).map(W65Register::Rw)).map(Register::new).collect()
    }

    fn add_assigns(
        &self,
        state: &Self::AssignParamsState<'_>,
        is_varargs: bool,
    ) -> Vec<(cmli::mach::Register, u64)> {
        Vec::new()
    }

    fn shadow_space(&self, state: &Self::AssignParamsState<'_>) -> u32 {
        0
    }

    fn redzone(&self, state: &Self::AssignParamsState<'_>) -> u32 {
        0
    }

    fn callee_cleanup_size(&self, state: &Self::AssignParamsState<'_>) -> u32 {
        0
    }

    fn stack_align(&self) -> (u32, u32, u32) {
        (4, 0 , 3)
    }
}

pub struct W65Compiler;

impl XvaCompiler for W65Compiler {
    fn call_conv(&self) -> &dyn crate::callconv::CallConv {
        const { &Spec::<W65CallConv>::new() }
    }

    fn machine_mode(&self) -> cmli::mach::MachineMode {
        MachineMode::new(W65Mode(0))
    }

    fn compiler(&self) -> &dyn cmli::compiler::Compiler {
        &W65{}
    }
}