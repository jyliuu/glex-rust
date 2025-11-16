use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{s, Array2};

use glex_core::{all_subsets, FastPD};

mod xgboost_bench_utils;
use xgboost_bench_utils::{
    load_features_from_csv, load_xgboost_trees, workspace_root,
    SYNTHETIC_LOW_DIM_STEMS,
};

fn benchmark_fastpd_eval_for_dataset(
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
    // Use the first up to 500 rows as evaluation points.
    let n_rows = background.nrows();
    let n_eval = n_rows.min(500);
    let eval_points: Array2<f32> = background.slice(s![0..n_eval, ..]).to_owned();

    let trees = load_xgboost_trees(&model_path);
    let n_features = background.ncols();
    let all_features = all_subsets(n_features);

    // Use the whole dataset as background for augmentation (preprocessing step).
    let background_view = background.view();
    let mut fastpd = FastPD::new(trees, &background_view, 0.0).expect("Failed to construct FastPD");

    let eval_view = eval_points.view();

    group.bench_function(BenchmarkId::from_parameter(dataset_stem), move |b| {
        b.iter(|| {
            for features in &all_features {
                let result = fastpd
                    .pd_function(&eval_view, features)
                    .expect("FastPD evaluation failed");
                black_box(result);
            }
        });
    });
}

fn fastpd_xgboost_eval_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("fastpd_xgboost_eval_500_random_points");

    // Synthetic datasets of increasing dimensionality
    for stem in &SYNTHETIC_LOW_DIM_STEMS {
        benchmark_fastpd_eval_for_dataset(&mut group, stem);
    }

    group.finish();
}

criterion_group!(benches, fastpd_xgboost_eval_benchmarks);
criterion_main!(benches);
