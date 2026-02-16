use lxca::ir::test_files;
use lxca_cg::{
    x86_64::X86_64Compiler,
    xva::{XvaCompiler, lower_lxca},
};
use target_tuples::TargetRef;

fn main() {
    let compiler = X86_64Compiler;

    let mach = compiler.compiler().machine();

    let target_name = "x86_64-pc-linux-gnu";

    let target =
        lccc_targets::builtin::target::from_target(&TargetRef::parse(target_name)).unwrap();

    let xva = lxca::ir::with_context(|ctx| {
        println!("return 42 test");
        let file = test_files::return_42(target_name, ctx);
        println!("lxca:");
        println!("{file:#?}");

        lower_lxca(&file, &target, &compiler)
    });

    println!("xva:");
    println!("{}", xva.pretty_print(mach));

    let xva = lxca::ir::with_context(|ctx| {
        println!("addition test");
        let file = test_files::addition(target_name, ctx);
        println!("lxca:");
        println!("{file:#?}");

        lower_lxca(&file, &target, &compiler)
    });

    println!("xva:");
    println!("{}", xva.pretty_print(mach));
}
