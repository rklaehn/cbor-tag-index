use fnv::{FnvHashMap, FnvHashSet};
use serde::{Deserialize, Serialize};
use std::{
    convert::TryFrom,
    fmt,
    ops::{Add, Sub},
};

#[cfg(test)]
extern crate quickcheck;
#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(into = "serde_bytes::ByteBuf", from = "serde_bytes::ByteBuf")]
pub struct Tag(Vec<u8>);

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
        Self(value.into_vec())
    }
}

/// A compact representation of a set of tag sets
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(into = "TagSetSetIo", try_from = "TagSetSetIo")]
pub struct TagSetSet {
    tags: Box<[Tag]>,
    sets: Box<[u128]>,
}

impl TagSetSet {
    pub fn iter<'a>(&'a self) -> TagSetSetIter<'a> {
        TagSetSetIter(&self.tags, self.sets.iter())
    }

    pub fn map_into(&self, target: &TagSetSet) -> Box<[u128]> {
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
        let result = self.sets.iter().filter_map(|mask| {
            mask_from_bits_iter(OneBitsIterator(*mask).map(|index| translate[index as usize])).ok()
        }).collect();
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
            .collect::<Vec<_>>()
            .into();
        let mut sets: Box<[u128]> = (0..self.sets.len())
            .map(|_| u128::default())
            .collect::<Vec<_>>()
            .into();
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
        let sets = value
            .1
            .iter_mut()
            .map(|index_deltas| {
                delta_decode::<u8>(index_deltas);
                mask_from_bits_iter(index_deltas.iter().map(|index| *index as u32))
            })
            .collect::<anyhow::Result<Vec<u128>>>()?
            .into();
        Ok(Self {
            sets,
            tags: value.0,
        })
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

impl TagSetSet {}

fn tags(tags: &[&str]) -> TagSet {
    tags.iter()
        .map(|x| Tag(x.to_string().into_bytes()))
        .collect()
}

fn main() -> anyhow::Result<()> {
    let a = tags(&["a", "b", "x", "y"]);
    let b = tags(&["a", "x", "y"]);
    let c = tags(&["a", "b", "y"]);
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
    use super::*;

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
}
