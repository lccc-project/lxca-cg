use std::collections::HashMap;

use cmli::{compiler::CompilerContext, instr::AddressKind, intern::Symbol, target::{PropertyValue, TargetInfo, TargetProperties}};
use lccc_targets::properties::{arch::Machine, target::Target};

use crate::xva::XvaCompiler;


bitflags::bitflags! {
    #[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
    pub struct CgFlags : u32 {
        const PIC = 0x0000_0001;
        const STATIC_PIE = 0x0000_0002;
        const NO_PLT = 0x0000_0004;
        const TLS_DYNAMIC = 0x0001_0001;
        const TLS_INIT_EXEC = 0x0001_0002;
    }
}
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct AddressKinds {
    pub global: AddressKind,
    pub global_call: AddressKind,
    pub local: AddressKind,
    pub tls: AddressKind,
    pub local_tls: AddressKind,
}


pub fn create_context(compiler: &dyn XvaCompiler, target: &Target, cg_flags: CgFlags, mach: Option<&Machine>) -> CompilerContext {
    let AddressKinds { global, global_call, local, tls, local_tls } = compiler.address_kinds(cg_flags);

    let mut context = CompilerContext {
        mode: compiler.machine_mode(),
        properties: TargetInfo {
            properties: TargetProperties {
                global_properties: target.compile_default_properties(mach).into_iter().map(|(k, v)| {
                    let key = Symbol::intern(k.as_ref());

                    let value = match v {
                        lccc_targets::properties::ExtPropertyValue::String(val) => PropertyValue::String(Symbol::intern(val.as_ref())),
                        lccc_targets::properties::ExtPropertyValue::Bool(val) => PropertyValue::Bool(val),
                        lccc_targets::properties::ExtPropertyValue::Int(val) => PropertyValue::Int(val),
                    };

                    (key, value)
                }).collect(),
            },
            ptr_width: target.primitive_layout.int_layout.short_pointer_width,
        },
        property_overrides: TargetProperties {
            global_properties: HashMap::new(),
        },
        target_features: target.compile_target_features(mach).into_iter().map(|v| v.to_box().into_string()).collect(),
        global_address_kind: global,
        global_call_address_kind: global_call,
        local_address_kind: local,
        global_tls_kind: tls,
        local_tls_kind: local_tls,
    };

    context
}