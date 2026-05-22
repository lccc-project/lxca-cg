use std::collections::HashSet;

use cmli::xva::{opt::{ALL_PASSES, run_passes}, regalloc::RegAllocator};
use lxca::ir::{
    self,
    test_files::{TEST_FILES},
};
use lxca_cg::{
    target::{CgFlags, create_context}, xva::{lower_lxca}
};
use target_tuples::TargetRef;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
enum Phase {
    Lxca,
    Xva,
    OptXva,
    Regalloc,
    Mce,
}

impl Phase {
    pub fn from_str(name: &str) -> Phase {
        match name {
            "lxca" => Phase::Lxca,
            "xva" => Phase::Xva,
            "optxva" => Phase::OptXva,
            "regalloc" => Phase::Regalloc,
            "mce" => Phase::Mce,
            _ => panic!("Unknown phase {name}")
        }
    }
}

fn main() {
    let mut args = std::env::args();
    let prg_name = args.next().unwrap();
    let target_name = args.next().unwrap_or_else(|| String::from("x86_64-pc-linux-gnu"));
    let target = TargetRef::parse(&target_name);

    let compiler = lxca_cg::xva::compiler_from_target(target).unwrap();

    let mach = compiler.compiler().machine();

    let mode = compiler.machine_mode();

    let target =
        lccc_targets::builtin::target::from_target(&target).unwrap();

    let mut global_feature = target.compile_target_features(None);

    let (print_phases, last_phase) = match args.next() {
        Some(phases) => {
            let mut max = Phase::Lxca;
            (phases.split(',').map(Phase::from_str).inspect(|v| max = max.max(*v)).collect::<HashSet<_>>(), max)
        }
        None => {
            ([Phase::Lxca, Phase::Mce].into_iter().collect::<HashSet<_>>(), Phase::Mce)
        }
    };

    let tests = args.collect::<HashSet<_>>();

    for &(name, test) in TEST_FILES {
        if !(tests.is_empty() || tests.contains(name)) {
            continue
        }
        println!("{name}:");
        let mut file = ir::with_context(|ctx| {
            let file = test(&target_name, ctx);

            if print_phases.contains(&Phase::Lxca) {
                println!("lxca ir:");
                println!("{file}");
            }

            lower_lxca(&file, &target, &global_feature, compiler)
        });

        if last_phase < Phase::Xva {
            continue
        }

        if print_phases.contains(&Phase::Xva) {
            println!("xva:");
            println!("{}", file.pretty_print(mach, mode));
        }
        
        if last_phase < Phase::OptXva {
            continue
        }

        let fuel = 100; // Treat this as a debug build, use 100 for now

        
        run_passes(
            ALL_PASSES.iter().copied(),
            cmli::xva::opt::XvaOptPhase::AfterLower,
            &mut file,
            fuel,
            mach,
            mode
        );

        if print_phases.contains(&Phase::OptXva) {
            println!("optimized xva:");
            println!("{}", file.pretty_print(mach, mode));
        }


        if last_phase < Phase::Regalloc {
            continue
        }
        run_passes(
            ALL_PASSES.iter().copied(),
            cmli::xva::opt::XvaOptPhase::BeforeRegalloc,
            &mut file,
            fuel / 2,
            mach,
            mode
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
            mach,
            mode
        );

        if print_phases.contains(&Phase::Regalloc) {
            println!("regalloc xva:");
            println!("{}", file.pretty_print(mach, mode));
        }

        if last_phase < Phase::Mce {
            continue
        }

        run_passes(
            ALL_PASSES.iter().copied(),
            cmli::xva::opt::XvaOptPhase::BeforeMce,
            &mut file,
            fuel / 2,
            mach,
            mode,
        );

        let context = create_context(compiler, &target, CgFlags::empty(), None);

        file.lower_mc(compiler.compiler(), &context);

        if print_phases.contains(&Phase::Mce) {
            println!("mce:");
            println!("{}", file.pretty_print(mach, mode));
        }


    }
}
