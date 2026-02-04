use cmli::{
    archs::x86::{X86Register, XmmSize},
    compiler::CompilerContext,
    target::PropertyValue,
    x86_registers,
    xva::XvaRegister,
};

use crate::callconv::{CallConvSpec, ParameterFragmentClass, StackOrder};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum X86_64Abi {
    SysV,
    Msabi,
    Vectorcall,
}

impl X86_64Abi {
    pub fn gregs(&self) -> &'static [X86Register] {
        match self {
            Self::SysV => &x86_registers!(rdi, rsi, rdx, rcx, r8, r9),
            Self::Msabi | Self::Vectorcall => &x86_registers!(rcx, rdx, r8, r9),
        }
    }

    pub fn allow_x87_return(&self) -> bool {
        matches!(self, Self::SysV)
    }

    pub fn vregs(&self) -> &'static [X86Register] {
        match self {
            Self::SysV => &x86_registers!(xmm0, xmm1, xmm2, xmm3, xmm4, xmm4, xmm5, xmm6, xmm7),
            Self::Msabi => &x86_registers!(xmm0, xmm1, xmm2, xmm3),
            Self::Vectorcall => &x86_registers!(xmm0, xmm1, xmm2, xmm3, xmm4, xmm4, xmm5),
        }
    }

    pub fn interleave_params(&self) -> bool {
        matches!(self, Self::SysV)
    }

    pub fn max_param_frags(&self) -> usize {
        match self {
            Self::SysV => 2,
            _ => 1,
        }
    }

    pub fn max_sseup_frags(&self) -> usize {
        match self {
            Self::SysV | Self::Vectorcall => 8,
            _ => 1,
        }
    }
}

#[derive(Copy, Clone)]
pub struct X86_64AbiState<'a> {
    ctx: &'a CompilerContext,
    vreg_pos: usize,
    greg_pos: usize,
    mark: Option<(usize, usize)>,
}

impl<'a> X86_64AbiState<'a> {
    pub fn mark(&mut self) {
        self.mark = Some((self.vreg_pos, self.greg_pos));
    }

    pub fn rewind(&mut self) {
        (self.vreg_pos, self.greg_pos) = self.mark.unwrap();
    }
    pub fn next_greg(&mut self, abi: X86_64Abi) -> Option<X86Register> {
        let reg = abi.gregs().get(self.greg_pos).copied()?;

        self.greg_pos += 1;
        if !abi.interleave_params() {
            self.vreg_pos += 1;
        }

        Some(reg)
    }

    pub fn next_vreg(&mut self, abi: X86_64Abi) -> Option<X86Register> {
        let reg = abi.vregs().get(self.vreg_pos).copied()?;

        self.vreg_pos += 1;
        if !abi.interleave_params() {
            self.vreg_pos += 1;
        }

        Some(reg)
    }
}

impl CallConvSpec for X86_64Abi {
    type AssignParamsState<'a> = X86_64AbiState<'a>;

    fn from_name(name: &str, ctx: &cmli::compiler::CompilerContext) -> Option<Self>
    where
        Self: Sized,
    {
        match name {
            "C" => match ctx.property("lxca.default_tag") {
                Some(PropertyValue::String(v)) => Self::from_name(v, ctx),
                _ => None,
            },
            "system" => match ctx.property("lxca.system_tag") {
                Some(PropertyValue::String(v)) => Self::from_name(v, ctx),
                _ => None,
            },
            "sysv64" => Some(Self::SysV),
            "msabi64" => Some(Self::Msabi),
            "vectorcall" => Some(Self::Vectorcall),
            _ => None,
        }
    }

    fn make_state(ctx: &cmli::compiler::CompilerContext) -> Self::AssignParamsState<'_>
    where
        Self: Sized,
    {
        X86_64AbiState {
            ctx,
            vreg_pos: 0,
            greg_pos: 0,
            mark: None,
        }
    }

    fn classify_int<F: FnMut(crate::callconv::ParameterFragmentClass, u32, u32)>(
        &self,
        bits: u16,
        _: &cmli::compiler::CompilerContext,
        mut v: F,
    ) {
        let total_fragments = (bits + 63) >> 6;
        let mut total_len = (bits + 7) >> 3;
        for x in 0..total_fragments {
            let frag_base = (x * 8) as u32;
            let len = total_len.min(8);
            v(ParameterFragmentClass::Integer, frag_base, len as u32);
            total_len -= len;
        }
    }

    fn stack_order(&self) -> crate::callconv::StackOrder {
        StackOrder::RightToLeft
    }

    fn assign_registers_param(
        &self,
        frags: &[ParameterFragmentClass],
        state: &mut Self::AssignParamsState<'_>,
        _: bool,
    ) -> Option<Vec<cmli::xva::XvaRegister>> {
        match frags {
            [ParameterFragmentClass::Integer] => {
                let reg = state.next_greg(*self)?;

                Some(vec![XvaRegister::physical(reg)])
            }
            [
                ParameterFragmentClass::Integer,
                ParameterFragmentClass::Integer,
            ] => {
                state.mark();
                let (Some(reg1), Some(reg2)) = (state.next_greg(*self), state.next_greg(*self))
                else {
                    state.rewind();
                    return None;
                };

                Some(vec![
                    XvaRegister::physical(reg1),
                    XvaRegister::physical(reg2),
                ])
            }
            [ParameterFragmentClass::Vector] => {
                let reg = state.next_vreg(*self)?;

                Some(vec![XvaRegister::physical(reg)])
            }
            [
                ParameterFragmentClass::Vector,
                ParameterFragmentClass::Extension,
                ..,
            ] => {
                let xmm = state.next_vreg(*self)?;
                let sz = match frags.len() {
                    2 => XmmSize::Xmm,
                    3 | 4 => XmmSize::Ymm,
                    5..8 => XmmSize::Zmm,
                    _ => unreachable!(),
                };

                let reg = xmm.promote_xmm(sz);

                Some(vec![XvaRegister::physical(reg)])
            }
            [
                ParameterFragmentClass::Vector,
                ParameterFragmentClass::Vector,
            ] => {
                state.mark();
                let (Some(reg1), Some(reg2)) = (state.next_vreg(*self), state.next_vreg(*self))
                else {
                    state.rewind();
                    return None;
                };

                Some(vec![
                    XvaRegister::physical(reg1),
                    XvaRegister::physical(reg2),
                ])
            }
            [
                ParameterFragmentClass::Integer,
                ParameterFragmentClass::Vector,
            ] => {
                state.mark();
                let (Some(reg1), Some(reg2)) = (state.next_greg(*self), state.next_vreg(*self))
                else {
                    state.rewind();
                    return None;
                };
                Some(vec![
                    XvaRegister::physical(reg1),
                    XvaRegister::physical(reg2),
                ])
            }
            [
                ParameterFragmentClass::Vector,
                ParameterFragmentClass::Integer,
            ] => {
                state.mark();
                let (Some(reg1), Some(reg2)) = (state.next_vreg(*self), state.next_greg(*self))
                else {
                    state.rewind();
                    return None;
                };
                Some(vec![
                    XvaRegister::physical(reg1),
                    XvaRegister::physical(reg2),
                ])
            }
            _ => unreachable!(),
        }
    }

    fn replace_with_memory_param(
        &self,
        frags: &[ParameterFragmentClass],
    ) -> Option<ParameterFragmentClass> {
        // ScalarFloat is used for long-double only
        if frags.contains(&ParameterFragmentClass::Memory)
            || frags.contains(&ParameterFragmentClass::ScalarFloat)
        {
            return Some(ParameterFragmentClass::Integer);
        }

        if frags.len() > self.max_param_frags() {
            match frags {
                [ParameterFragmentClass::Vector, rest @ ..]
                    if frags.len() <= self.max_sseup_frags() =>
                {
                    if !rest
                        .iter()
                        .all(|frag| *frag == ParameterFragmentClass::Extension)
                    {
                        return Some(ParameterFragmentClass::Integer);
                    } else {
                        return None;
                    }
                }
                _ => return Some(ParameterFragmentClass::Integer),
            }
        }

        None
    }

    fn replace_with_memory_return(
        &self,
        frags: &[ParameterFragmentClass],
    ) -> Option<ParameterFragmentClass> {
        // Special-case st(0), st(1)
        match frags {
            [
                ParameterFragmentClass::ScalarFloat,
                ParameterFragmentClass::Extension,
            ] if self.allow_x87_return() => return None,
            [
                ParameterFragmentClass::ScalarFloat,
                ParameterFragmentClass::Extension,
                ParameterFragmentClass::Extension,
                ParameterFragmentClass::Extension,
            ] if self.allow_x87_return() => return None,
            _ => {}
        }
        // ScalarFloat is used for long-double only
        if frags.contains(&ParameterFragmentClass::Memory)
            || frags.contains(&ParameterFragmentClass::ScalarFloat)
        {
            return Some(ParameterFragmentClass::Integer);
        }

        if frags.len() > self.max_param_frags() {
            match frags {
                [ParameterFragmentClass::Vector, rest @ ..]
                    if frags.len() <= self.max_sseup_frags() =>
                {
                    if !rest
                        .iter()
                        .all(|frag| *frag == ParameterFragmentClass::Extension)
                    {
                        return Some(ParameterFragmentClass::Integer);
                    } else {
                        return None;
                    }
                }
                _ => return Some(ParameterFragmentClass::Integer),
            }
        }

        None
    }

    fn assign_registers_return(
        &self,
        frags: &[ParameterFragmentClass],
    ) -> Vec<cmli::xva::XvaRegister> {
        let mut regs = Vec::with_capacity(frags.len().min(2));

        // Special Cases
        match frags {
            [
                ParameterFragmentClass::Vector,
                ParameterFragmentClass::Extension,
                ..,
            ] => {
                let xmm = X86Register::Xmm(0);
                let sz = match frags.len() {
                    2 => XmmSize::Xmm,
                    3 | 4 => XmmSize::Ymm,
                    5..8 => XmmSize::Zmm,
                    _ => unreachable!(),
                };

                let reg = xmm.promote_xmm(sz);

                regs.push(XvaRegister::physical(reg))
            }
            [
                ParameterFragmentClass::ScalarFloat,
                ParameterFragmentClass::Extension,
            ] => {
                regs.push(XvaRegister::physical(X86Register::St(0)));
            }
            [
                ParameterFragmentClass::ScalarFloat,
                ParameterFragmentClass::Extension,
                ParameterFragmentClass::Extension,
                ParameterFragmentClass::Extension,
            ] => {
                regs.push(XvaRegister::physical(X86Register::St(0)));
                regs.push(XvaRegister::physical(X86Register::St(1)));
            }
            frags => {
                for (i, frag) in frags.iter().enumerate() {
                    match frag {
                        ParameterFragmentClass::Integer => {
                            regs.push(XvaRegister::physical(x86_registers![rax, rdx][i]))
                        }

                        ParameterFragmentClass::Vector => {
                            regs.push(XvaRegister::physical(x86_registers![xmm0, xmm1][i]))
                        }
                        ParameterFragmentClass::ScalarFloat
                        | ParameterFragmentClass::Extension
                        | ParameterFragmentClass::Memory
                        | ParameterFragmentClass::Other(_) => unreachable!(),
                    }
                }
            }
        }

        regs
    }

    fn return_return_place(
        &self,
        frag: &[ParameterFragmentClass],
    ) -> Option<ParameterFragmentClass> {
        None // return place is never returned
    }

    fn non_volatile_registers(&self) -> Vec<XvaRegister> {
        let regs: &[X86Register] = match self {
            X86_64Abi::SysV => &x86_registers![rbx, rbp, r12, r13, r14, r15],
            X86_64Abi::Msabi | X86_64Abi::Vectorcall => {
                &x86_registers![rbx, rbp, rsi, di, r12, r13, r14, r15]
            }
        };

        regs.iter()
            .copied()
            .map(|v| XvaRegister::physical(v))
            .collect()
    }

    fn volatile_registers(&self) -> Vec<XvaRegister> {
        let regs: &[X86Register] = match self {
            X86_64Abi::SysV => &x86_registers![
                rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11, r16, r17, r18, r19, r20, r21, r22, r23,
                r24, r25, r26, r27, r28, r29, r30, r31, xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6,
                xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15, xmm16, xmm17, xmm18,
                xmm19, xmm20, xmm21, xmm22, xmm23, xmm24, xmm25, xmm26, xmm27, xmm28, xmm29, xmm30,
                xmm31, tmm0, tmm1, tmm2, tmm3, tmm4, tmm5, tmm6, tmm7, k0, k1, k2, k3, k4, k5, k6,
                k7, st0, st1, st2, st3, st4, st5, st6, st7,
            ],
            X86_64Abi::Msabi | X86_64Abi::Vectorcall => &x86_registers![
                rax, rcx, rdx, r8, r9, r10, r11, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25,
                r26, r27, r28, r29, r30, r31, xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8,
                xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15, xmm16, xmm17, xmm18, xmm19, xmm20,
                xmm21, xmm22, xmm23, xmm24, xmm25, xmm26, xmm27, xmm28, xmm29, xmm30, xmm31, tmm0,
                tmm1, tmm2, tmm3, tmm4, tmm5, tmm6, tmm7, k0, k1, k2, k3, k4, k5, k6, k7, st0, st1,
                st2, st3, st4, st5, st6, st7,
            ],
        };

        regs.iter()
            .copied()
            .map(|v| XvaRegister::physical(v))
            .collect()
    }

    fn add_assigns(&self, state: &X86_64AbiState<'_>, is_varargs: bool) -> Vec<(XvaRegister, u64)> {
        match self {
            X86_64Abi::SysV => {
                if is_varargs {
                    vec![(
                        XvaRegister::physical(X86Register::Byte(0)),
                        state.vreg_pos as u64,
                    )]
                } else {
                    Vec::new()
                }
            }
            X86_64Abi::Msabi | X86_64Abi::Vectorcall => Vec::new(),
        }
    }

    fn shadow_space(&self, _state: &Self::AssignParamsState<'_>) -> u32 {
        match self {
            X86_64Abi::SysV => 0,
            X86_64Abi::Msabi | X86_64Abi::Vectorcall => 32,
        }
    }

    fn redzone(&self, state: &Self::AssignParamsState<'_>) -> u32 {
        match self {
            X86_64Abi::SysV => match state.ctx.property("x86.abi.enable-redzone") {
                Some(PropertyValue::Bool(true)) | None => 128,
                Some(_) => 0,
            },
            X86_64Abi::Msabi | X86_64Abi::Vectorcall => 0,
        }
    }

    fn callee_cleanup_size(&self, _state: &Self::AssignParamsState<'_>) -> u32 {
        0
    }
}
