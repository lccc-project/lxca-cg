use std::range::{Range, RangeInclusive, RangeInclusiveIter};

use cmli::{archs::skyarch::{Map, Skyarch, SkyarchCoprocessor, SkyarchRegRangeIter, SkyarchRegister, SkyarchRegno}, as_id_array, mach::{MachineMode, OneMachine, Register}, skyarch_regno, traits::IdType};
use lccc_targets::properties::target::Target;

use crate::{callconv::{CallConvSpec, ParameterFragmentClass, Spec, StackOrder}, xva::XvaCompiler};

pub struct SkyarchCoprocCc {
    pub coproc_num: SkyarchCoprocessor,
    pub regno_range: RangeInclusive<SkyarchRegno>,
}

pub struct SkyarchCc {
    use_hardfp: Option<SkyarchCoprocCc>,
    use_hardvreg: Option<SkyarchCoprocCc>,
    ireg_range: RangeInclusive<SkyarchRegno>,
    iret_regs: RangeInclusive<SkyarchRegno>
}


pub struct SkyarchAssignParamsState<'a> {
    target: &'a Target,
    ireg_pos: u32,
}

impl CallConvSpec for SkyarchCc {    
    type AssignParamsState<'a> = SkyarchAssignParamsState<'a>;
    
    fn from_name(name: &str, _: &lccc_targets::properties::target::Target) -> Option<Self>
    where
        Self: Sized {
        match name {
            "C" => Some(SkyarchCc { use_hardfp: None, use_hardvreg: None, ireg_range: RangeInclusive { start: skyarch_regno!(1), last: skyarch_regno!(10) }, iret_regs: RangeInclusive { start: skyarch_regno!(1), last: skyarch_regno!(4) } }),
            _ => None,
        }
    }
    
    fn make_state(ctx: &lccc_targets::properties::target::Target) -> Self::AssignParamsState<'_>
    where
        Self: Sized {
        SkyarchAssignParamsState {
            target: ctx,
            ireg_pos: 0,
        }
    }
    
    fn classify_int<F: FnMut(crate::callconv::ParameterFragmentClass, u32, u32)>(
        &self,
        bits: u16,
        _: &lccc_targets::properties::target::Target,
        mut v: F,
    ) {
        let mut bytes = ((bits + 7) >> 3) as u32;
        let mut pos = 0;
        while bytes > 0 {
            let width = bytes.clamp(0, 4);

            v(ParameterFragmentClass::Integer, pos, width);

            pos += width;
            bytes -= width;
        }
    }
    
    fn stack_order(&self) -> crate::callconv::StackOrder {
        StackOrder::RightToLeft
    }
    
    fn replace_with_memory_param(
        &self,
        frags: &[(crate::callconv::ParameterFragmentClass, u32)],
    ) -> Option<(crate::callconv::ParameterFragmentClass, u32)> {
        if frags.len() > 4 {
            return Some((ParameterFragmentClass::Integer, 4))
        } 

        if frags.iter().any(|(a, _)| *a == ParameterFragmentClass::Memory) {
            return Some((ParameterFragmentClass::Integer, 4)) 
        }

        None
    }
    
    fn replace_with_memory_return(
        &self,
        frags: &[(crate::callconv::ParameterFragmentClass, u32)],
    ) -> Option<(crate::callconv::ParameterFragmentClass, u32)> {
       self.replace_with_memory_param(frags) // Same Logic used on both paths
    }
    
    fn assign_registers_param(
        &self,
        frags: &[(crate::callconv::ParameterFragmentClass, u32)],
        state: &mut Self::AssignParamsState<'_>,
        _: bool,
    ) -> Option<Vec<cmli::mach::Register>> {
        let regs = SkyarchRegRangeIter::from_range(self.iret_regs);

        let ret: Vec<_> = regs.zip(frags)
            .skip(state.ireg_pos as usize)
            .map(|(r, _)| Map::GeneralPurpose.reg_in(r))
            .map(Register::new)
            .collect();

        if ret.len() < frags.len() {
            return None
        }

        state.ireg_pos += ret.len() as u32;
        Some(ret)
    }
    
    fn assign_registers_return(&self, frags: &[(crate::callconv::ParameterFragmentClass, u32)]) -> Vec<cmli::mach::Register> {
        let regs = SkyarchRegRangeIter::from_range(self.iret_regs);

        regs.zip(frags)
            .map(|(r, _)| Map::GeneralPurpose.reg_in(r))
            .map(Register::new)
            .collect()
    }
    
    fn return_return_place(
        &self,
        frag: &[(crate::callconv::ParameterFragmentClass, u32)],
    ) -> Option<(crate::callconv::ParameterFragmentClass, u32)> {
        None
    }
    
    fn volatile_registers(&self) -> Vec<cmli::mach::Register> {
        core::iter::repeat_n(1, 15)
            .enumerate()
            .map(|(a, b)| (a as u64) + b)
            .map(SkyarchRegister)
            .map(Register::new)
            .chain(core::iter::repeat_n(2 << 5, 32)
                .enumerate()
                .map(|(a, b)| (a as u64) + b)
                .map(SkyarchRegister)
                .map(Register::new)
            )
            .collect()
    }
    
    fn non_volatile_registers(&self) -> Vec<cmli::mach::Register> {
        core::iter::repeat_n(16, 12)
            .enumerate()
            .map(|(a, b)| (a as u64) + b)
            .map(SkyarchRegister)
            .map(Register::new)
            .collect()
    }
    
    fn add_assigns(
        &self,
        state: &Self::AssignParamsState<'_>,
        is_varargs: bool,
    ) -> Vec<(cmli::mach::Register, u64)> {
        vec![]
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
        (4, 0, 0)
    } 
}

pub struct SkyarchCompiler;

impl XvaCompiler for SkyarchCompiler {
    fn call_conv(&self) -> &dyn crate::callconv::CallConv {
        const {&Spec::<SkyarchCc>::new() }
    }

    fn machine_mode(&self) -> cmli::mach::MachineMode {
        MachineMode::new(OneMachine::Singleton)
    }

    fn compiler(&self) -> &dyn cmli::compiler::Compiler {
        &Skyarch
    }
}