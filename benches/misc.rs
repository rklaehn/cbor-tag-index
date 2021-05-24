use cbor_tag_index::{DnfQuery, TagIndex, TagSet};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use libipld::codec::Codec;
use libipld_cbor::DagCborCodec;
use rand::{prelude::*, SeedableRng};
use rand_chacha::ChaChaRng;

fn create_example(
    seed: u64,
    n_events: usize,
    n_terms: usize,
) -> anyhow::Result<(TagIndex<String>, DnfQuery<String>)> {
    let mut rng = ChaChaRng::seed_from_u64(seed);
    let mut id = || -> String { hex::encode(rng.gen::<u64>().to_be_bytes()) };
    let mut create_concrete = |prefix: &str, n: usize| -> Vec<String> {
        (0..n).map(|_| format!("{}-{}", prefix, id())).collect()
    };
    let common: Vec<(&str, usize)> = vec![("article", 30), ("sku", 40), ("location", 50)];
    let tags: Vec<(String, Vec<String>)> = common
        .into_iter()
        .map(|(prefix, n)| (prefix.to_owned(), create_concrete(prefix, n)))
        .collect();
    let events = (0..n_events)
        .filter_map(|_| {
            let (common, rare) = tags.choose(&mut rng).unwrap().clone();
            let rare = rare.choose(&mut rng)?.clone();
            Some(vec![common, rare].into_iter().collect::<TagSet<String>>())
        })
        .collect::<Vec<_>>();
    let query = (0..n_terms)
        .filter_map(|_| {
            let (common, rare) = tags.choose(&mut rng).unwrap().clone();
            let rare = rare.choose(&mut rng)?.clone();
            Some(vec![common, rare].into_iter().collect::<TagSet<String>>())
        })
        .collect::<Vec<_>>();
    let index = TagIndex::new(&events)?;
    let query = DnfQuery::new(&query)?;
    Ok((index, query))
}

fn encode(index: &TagIndex<String>) -> anyhow::Result<Vec<u8>> {
    DagCborCodec.encode(index)
}

fn decode(bytes: &[u8]) -> anyhow::Result<TagIndex<String>> {
    DagCborCodec.decode(bytes)
}

fn criterion_benchmark(c: &mut Criterion) {
    let (index, query) = create_example(0, 10000, 3).unwrap();
    let bytes = DagCborCodec.encode(&index).unwrap();
    c.bench_function("decode", |b| b.iter(|| decode(black_box(&bytes))));
    c.bench_function("encode", |b| b.iter(|| encode(black_box(&index))));
    c.bench_function("filter", |b| b.iter(|| query.matching(black_box(&index))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
