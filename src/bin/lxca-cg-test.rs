use lxca::ir::test_files;
use lxca_cg::{x86_64::X86_64Compiler, xva::lower_lxca};
use target_tuples::TargetRef;

fn main() {
    let compiler = X86_64Compiler;

    let target_name = "x86_64-pc-linux-gnu";

    let target =
        lccc_targets::builtin::target::from_target(&TargetRef::parse(target_name)).unwrap();

    let xva = lxca::ir::with_context(|ctx| {
        let file = test_files::return_42(target_name, ctx);
        println!("lxca:");
        println!("{file:#?}");

        lower_lxca(&file, &target, &compiler)
    });

    println!("xva:");
    println!("{xva:#?}");
}
