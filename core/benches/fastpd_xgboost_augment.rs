use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use glex_core::FastPD;

mod xgboost_bench_utils;
use xgboost_bench_utils::{
    load_features_from_csv, load_xgboost_trees, workspace_root, CALIFORNIA_STEM, SYNTHETIC_STEMS,
};

fn benchmark_fastpd_augment_for_dataset(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    dataset_stem: &str,
) {
    let root = workspace_root();
    let data_dir = root.join("data");
    let datasets_dir = data_dir.join("datasets");
    let models_dir = data_dir.join("models");

    let csv_path = datasets_dir.join(format!("{dataset_stem}.csv"));
    let model_path = models_dir.join(format!("{dataset_stem}_xgb.json"));

    let background = load_features_from_csv(&csv_path);
    let trees = load_xgboost_trees(&model_path);

    let background_view = background.view();

    group.bench_function(BenchmarkId::from_parameter(dataset_stem), |b| {
        b.iter(|| {
            // Clone trees so each iteration re-runs augmentation from scratch.
            let trees_clone = trees.clone();
            let fastpd = FastPD::new(trees_clone, &background_view, 0.0)
                .expect("Failed to construct FastPD");
            black_box(fastpd.num_trees());
        });
    });
}

fn fastpd_xgboost_augment_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("fastpd_xgboost_augment");

    // Synthetic datasets of increasing dimensionality
    for stem in &SYNTHETIC_STEMS {
        benchmark_fastpd_augment_for_dataset(&mut group, stem);
    }

    // California housing dataset
    benchmark_fastpd_augment_for_dataset(&mut group, CALIFORNIA_STEM);

    group.finish();
}

criterion_group!(benches, fastpd_xgboost_augment_benchmarks);
criterion_main!(benches);
