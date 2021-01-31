use fmt::{Display, Write};
use fnv::{FnvHashMap, FnvHashSet};
use serde::{Deserialize, Serialize};
use std::{
    collections::binary_heap::Iter,
    convert::TryFrom,
    fmt,
    iter::FromIterator,
    mem::swap,
    ops::{Add, Sub},
};

#[cfg(test)]
extern crate quickcheck;
#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(into = "serde_bytes::ByteBuf", from = "serde_bytes::ByteBuf")]
pub struct Tag(Box<[u8]>);

impl fmt::Display for Tag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Ok(text) = std::str::from_utf8(&self.0) {
            write!(f, "{}", text)
        } else {
            write!(f, "{}", hex::encode(&self.0))
        }
    }
}

type TagSet = FnvHashSet<Tag>;

impl From<Tag> for serde_bytes::ByteBuf {
    fn from(value: Tag) -> Self {
        Self::from(value.0)
    }
}

impl From<serde_bytes::ByteBuf> for Tag {
    fn from(value: serde_bytes::ByteBuf) -> Self {
        Self(value.into_vec().into())
    }
}

/// A compact representation of a set of tag sets
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(into = "TagSetSetIo", try_from = "TagSetSetIo")]
pub struct TagSetSet {
    tags: Box<[Tag]>,
    sets: Box<[u128]>,
}

impl Display for TagSetSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set()
            .entries(self.iter().map(|x| x.collect::<DebugUsingDisplay<_>>()))
            .finish()
    }
}

struct DebugUsingDisplay<T>(Vec<T>);

impl<T: Display> fmt::Debug for DebugUsingDisplay<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_char('{')?;
        for (i, x) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_char(',')?;
            }
            Display::fmt(x, f)?;
        }
        f.write_char('}')?;
        Ok(())
    }
}

impl<T> FromIterator<T> for DebugUsingDisplay<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(Vec::from_iter(iter))
    }
}

/// Same, but optimized for dnf queries
#[derive(Debug, Clone, Default)]
pub struct TagSetSet2 {
    tags: FnvHashMap<Tag, u8>,
    sets: Box<[u128]>,
}

/// Turns an std::slice::IterMut<T> into an interator of T provided T has a default
///
/// This makes sense for cases where cloning is expensive, but default is cheap. E.g. Vec<T>.
struct SliceIntoIter<'a, T>(std::slice::IterMut<'a, T>);

impl<'a, T: Default + 'a> Iterator for SliceIntoIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|x| {
            let mut r = T::default();
            swap(x, &mut r);
            r
        })
    }
}

impl From<TagSetSet> for TagSetSet2 {
    fn from(mut value: TagSetSet) -> Self {
        let tags: FnvHashMap<Tag, u8> = SliceIntoIter(value.tags.iter_mut())
            .enumerate()
            .map(|(index, tag)| (tag, index as u8))
            .collect();
        Self {
            tags,
            sets: value.sets,
        }
    }
}

impl TagSetSet {
    pub fn iter<'a>(&'a self) -> TagSetSetIter<'a> {
        TagSetSetIter(&self.tags, self.sets.iter())
    }

    pub fn dnf_query(&self, dnf: &TagSetSet) -> impl Iterator<Item = bool> + '_ {
        let dnf = dnf.map_into(self);
        self.sets
            .iter()
            .map(move |set| dnf.iter().any(move |query| is_subset(*query, *set)))
    }

    fn map_into(&self, target: &TagSetSet) -> Box<[u128]> {
        let lookup: FnvHashMap<&Tag, u64> = target
            .tags
            .iter()
            .enumerate()
            .map(|(index, tag)| (tag, index as u64))
            .collect();
        let translate = self
            .tags
            .iter()
            .map(|tag| lookup.get(tag).cloned().unwrap_or(128))
            .collect::<Box<_>>();
        let result = self
            .sets
            .iter()
            .filter_map(|mask| {
                mask_from_bits_iter(OneBitsIterator(*mask).map(|index| translate[index as usize]))
                    .ok()
            })
            .collect();
        result
    }

    fn map_into_2(&self, target: &TagSetSet2) -> Box<[u128]> {
        let translate = self
            .tags
            .iter()
            .map(|tag| target.tags.get(tag).cloned().unwrap_or(128))
            .collect::<Box<_>>();
        let result = self
            .sets
            .iter()
            .filter_map(|mask| {
                mask_from_bits_iter(OneBitsIterator(*mask).map(|index| translate[index as usize]))
                    .ok()
            })
            .collect();
        result
    }
}

pub struct TagSetSetIter<'a>(&'a [Tag], std::slice::Iter<'a, u128>);

impl<'a> Iterator for TagSetSetIter<'a> {
    type Item = TagRefIter<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.1
            .next()
            .map(|mask| TagRefIter(self.0, OneBitsIterator(*mask)))
    }
}

pub struct TagRefIter<'a>(&'a [Tag], OneBitsIterator);

impl<'a> Iterator for TagRefIter<'a> {
    type Item = &'a Tag;

    fn next(&mut self) -> Option<Self::Item> {
        self.1.next().map(|index| &self.0[index as usize])
    }
}

#[derive(Debug, Default)]
pub struct TagSetSetBuilder {
    indices: FnvHashMap<Tag, usize>,
    sets: FnvHashMap<u128, usize>,
    next_index: usize,
}

impl TagSetSetBuilder {
    pub fn result(self) -> TagSetSet {
        let mut tags: Box<[Tag]> = (0..self.indices.len())
            .map(|_| Tag::default())
            .collect::<Box<_>>();
        let mut sets: Box<[u128]> = (0..self.sets.len())
            .map(|_| u128::default())
            .collect::<Box<_>>();
        for (tag, index) in self.indices.into_iter() {
            tags[index] = tag;
        }
        for (mask, index) in self.sets.into_iter() {
            sets[index] = mask;
        }
        TagSetSet { tags, sets }
    }

    pub fn push(&mut self, tags: TagSet) -> anyhow::Result<usize> {
        let indices = tags.into_iter().map(|tag| self.add_tag(tag));
        let mask = mask_from_bits_iter(indices.map(|x| x as u32))?;
        let offset = self.sets.len();
        self.sets.insert(mask, offset);
        Ok(offset)
    }

    fn add_tag(&mut self, tag: Tag) -> usize {
        if let Some(index) = self.indices.get(&tag) {
            *index
        } else {
            let index = self.next_index;
            self.next_index += 1;
            self.indices.insert(tag, index);
            index
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TagSetSetIo(Box<[Tag]>, Box<[Box<[u8]>]>);

fn delta_encode<T: Sub<Output = T> + Copy>(data: &mut [T]) {
    for i in (1..data.len()).rev() {
        data[i] = data[i] - data[i - 1];
    }
}

fn delta_decode<T: Add<Output = T> + Copy>(data: &mut [u8]) {
    for i in 1..data.len() {
        data[i] = data[i] + data[i - 1];
    }
}

impl From<TagSetSet> for TagSetSetIo {
    fn from(value: TagSetSet) -> Self {
        let bits = value
            .sets
            .iter()
            .map(|set| {
                let mut indices = OneBitsIterator(*set)
                    .map(|index| index as u8)
                    .collect::<Box<_>>();
                delta_encode::<u8>(&mut indices);
                indices
            })
            .collect();
        Self(value.tags, bits)
    }
}

impl TryFrom<TagSetSetIo> for TagSetSet {
    type Error = anyhow::Error;

    fn try_from(mut value: TagSetSetIo) -> Result<Self, Self::Error> {
        let tags = value.0;
        let sets = value
            .1
            .iter_mut()
            .map(|index_deltas| {
                delta_decode::<u8>(index_deltas);
                // check that all indices are valid
                anyhow::ensure!(index_deltas.iter().all(|x| (*x as usize) < tags.len()));
                mask_from_bits_iter(index_deltas.iter().cloned())
            })
            .collect::<anyhow::Result<Box<_>>>()?;
        Ok(Self { sets, tags })
    }
}

fn is_subset(a: u128, b: u128) -> bool {
    a & b == a
}

fn intersects(a: u128, b: u128) -> bool {
    a & b != 0
}

pub struct OneBitsIterator(u128);

impl Iterator for OneBitsIterator {
    type Item = u32;
    fn next(&mut self) -> Option<Self::Item> {
        let offset = self.0.trailing_zeros();
        if offset == 128 {
            None
        } else {
            self.0 &= !(1u128 << offset);
            Some(offset)
        }
    }
}

/// Given an interator of bits, creates a 128 bit bitmask.
/// If any of the bits is too high, returns an error.
fn mask_from_bits_iter(iterator: impl IntoIterator<Item = impl Into<u64>>) -> anyhow::Result<u128> {
    let mut result: u128 = 0;
    for bit in iterator.into_iter() {
        let bit: u64 = bit.into();
        anyhow::ensure!(bit < 128);
        result |= 1u128 << bit;
    }
    Ok(result)
}

fn tag_set(tags: &[&str]) -> TagSet {
    tags.iter()
        .map(|x| Tag(x.to_string().into_bytes().into()))
        .collect()
}

fn tag_set_set(tags: &[&[&str]]) -> TagSetSet {
    let mut builder = TagSetSetBuilder::default();
    for tags in tags.iter().map(|x| tag_set(x)) {
        builder.push(tags).unwrap();
    }
    builder.result()
}

fn main() -> anyhow::Result<()> {
    let a = tag_set(&["a", "b", "x", "y"]);
    let b = tag_set(&["a", "x", "y"]);
    let c = tag_set(&["a", "b", "y"]);
    let mut builder = TagSetSetBuilder::default();
    builder.push(a)?;
    builder.push(b)?;
    builder.push(c)?;
    let t = builder.result();
    println!("{:?}", t);
    for x in t.iter() {
        let x = x.map(|t| t.to_string()).collect::<Vec<_>>();
        println!("{:?}", x);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use quickcheck::Arbitrary;

    use super::*;

    #[test]
    fn dnf_query() {
        {
            let tags = tag_set_set(&[&["a", "b"], &["a", "c"], &["b", "c"]]);
            let dnf = tag_set_set(&[&["a"]]);
            let indexes = tags.dnf_query(&dnf).collect::<Box<_>>();
            assert_eq!(indexes.as_ref(), &[true, true, false]);
        }
        {
            let tags = tag_set_set(&[&["a", "b"], &["a", "c"], &["b", "c"]]);
            let dnf = tag_set_set(&[&["a", "b"]]);
            let indexes = tags.dnf_query(&dnf).collect::<Box<_>>();
            assert_eq!(indexes.as_ref(), &[true, false, false]);
        }
        {
            let tags = tag_set_set(&[&["a", "b"], &["a", "c"], &["b", "c"]]);
            let dnf = tag_set_set(&[&["a", "x"]]);
            let indexes = tags.dnf_query(&dnf).collect::<Box<_>>();
            assert_eq!(indexes.as_ref(), &[false, false, false]);
        }
        {
            let tags = tag_set_set(&[&["a", "b"], &["a", "c"], &["b", "c"]]);
            let dnf = tag_set_set(&[&[]]);
            let indexes = tags.dnf_query(&dnf).collect::<Box<_>>();
            assert_eq!(indexes.as_ref(), &[true, true, true]);
        }
        {
            let tags = tag_set_set(&[&["a", "b"], &["a", "c"], &["b", "c"]]);
            let dnf = tag_set_set(&[&["c"]]);
            let indexes = tags.dnf_query(&dnf).collect::<Box<_>>();
            assert_eq!(indexes.as_ref(), &[false, true, true]);
        }
        {
            let tags = tag_set_set(&[&["a", "b"], &["b", "c"], &["c", "d"]]);
            let dnf = tag_set_set(&[&["a"], &["d"]]);
            let indexes = tags.dnf_query(&dnf).collect::<Box<_>>();
            assert_eq!(indexes.as_ref(), &[true, false, true]);
        }
    }

    #[quickcheck]
    fn bits_iter_roundtrip(value: u128) -> bool {
        let iter = OneBitsIterator(value);
        let value1 = mask_from_bits_iter(iter).unwrap();
        value == value1
    }

    #[quickcheck]
    fn delta_decode_roundtrip(mut values: Vec<u8>) -> bool {
        values.sort();
        values.dedup();
        let reference = values.clone();
        delta_encode(&mut values);
        delta_decode::<u8>(&mut values);
        values == reference
    }

    #[quickcheck]
    fn tag_set_set_cbor_roundtrip(value: TagSetSet) -> bool {
        let bytes = serde_cbor::to_vec(&value).unwrap();
        let value1 = serde_cbor::from_slice(&bytes).unwrap();
        value == value1
    }

    impl Arbitrary for Tag {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let tag = g.choose(TAG_NAMES).unwrap();
            Tag(tag.as_bytes().to_vec().into())
        }
    }

    impl Arbitrary for TagSetSet {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let tags: FnvHashSet<Tag> = Arbitrary::arbitrary(g);
            let mut sets: Vec<u128> = Arbitrary::arbitrary(g);
            let mut tags = tags.into_iter().collect::<Vec<_>>();
            tags.truncate(128);
            let mask = !(u128::max_value() << tags.len());
            sets.iter_mut().for_each(|set| *set &= mask);
            sets.sort();
            sets.dedup();
            Self {
                sets: sets.into(),
                tags: tags.into(),
            }
        }
    }

    const TAG_NAMES: &[&'static str] = &["a", "b", "c", "d", "e", "f"];
}
