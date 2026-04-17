use cmli::xva::{opt::{ALL_PASSES, run_passes}, regalloc::RegAllocator};
use lxca::ir::{
    self,
    test_files::{self, TEST_FILES},
};
use lxca_cg::{
    x86_64::X86_64Compiler,
    xva::{XvaCompiler, lower_lxca},
};
use target_tuples::TargetRef;

fn main() {
    let compiler = X86_64Compiler;

    let mach = compiler.compiler().machine();

    let target_name = "x86_64-pc-linux-gnu";
    let mode = compiler.machine_mode();

    let target =
        lccc_targets::builtin::target::from_target(&TargetRef::parse(target_name)).unwrap();

    for &(name, test) in TEST_FILES {
        println!("{name}:");
        let mut file = ir::with_context(|ctx| {
            let file = test(target_name, ctx);

            println!("lxca ir:");
            println!("{file}");

            lower_lxca(&file, &target, &compiler)
        });

        println!("xva:");
        println!("{}", file.pretty_print(mach, mode));

        let fuel = 100; // Treat this as a debug build, use 100 for now

        // println!("optimized xva:");
        run_passes(
            ALL_PASSES.iter().copied(),
            cmli::xva::opt::XvaOptPhase::AfterLower,
            &mut file,
            fuel,
        ); // treat this as debug build, use 100 for now
        // println!("{}", file.pretty_print(mach, mode));

        run_passes(
            ALL_PASSES.iter().copied(),
            cmli::xva::opt::XvaOptPhase::BeforeRegalloc,
            &mut file,
            fuel / 2,
        ); 
        for func in &mut file.functions {
            let mut regallocer = RegAllocator::new(compiler.compiler(), &mut func.body);

            regallocer.process_function();
        }

        run_passes(
            ALL_PASSES.iter().copied(),
            cmli::xva::opt::XvaOptPhase::AfterRegalloc,
            &mut file,
            fuel / 2,
        ); 

        println!("regalloc xva:");
        println!("{}", file.pretty_print(mach, mode));

        run_passes(
            ALL_PASSES.iter().copied(),
            cmli::xva::opt::XvaOptPhase::BeforeMce,
            &mut file,
            fuel / 2,
        );

        file.lower_mc(compiler.compiler(), compiler.machine_mode());

        println!("mce:");
        println!("{}", file.pretty_print(mach, mode));


    }
}
