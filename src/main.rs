use core::slice;
use fmt::{Display, Write};
use fnv::{FnvHashMap, FnvHashSet};
use libipld::{
    cbor::DagCborCodec,
    codec::{Decode, Encode},
    DagCbor,
};
use num_traits::{WrappingAdd, WrappingSub};
use std::{convert::TryFrom, fmt, iter::FromIterator, mem::swap, result};
use vec_collections::VecSet;

#[cfg(test)]
extern crate quickcheck;
#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

type IndexSet = VecSet<[u32; 4]>;
type IndexMask = u128;

#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, DagCbor)]
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

/// A compact representation of a seq of tag sets
#[derive(Debug, Clone, Default, PartialEq, Eq, DagCbor)]
pub struct DnfQuery {
    tags: Vec<Tag>,
    sets: Bitmap,
}

impl DnfQuery {
    fn set_matching(&self, index: TagIndex, matches: &mut [bool]) {
        // create a bool array corresponding to the distinct tagsets in the index
        let mut tmp = vec![false; index.tags.sets.len()];
        // set the fields we need to look at to true
        for (matching, index) in matches.iter().zip(index.events.iter()) {
            if *matching {
                tmp[*index as usize] = true;
            }
        }
        // evaluate the dnf query for these fields
        index.tags.dnf_query(&self, &mut tmp);
        // write result from tmp
        for (matching, index) in matches.iter_mut().zip(index.events.iter()) {
            *matching = *matching && tmp[*index as usize];
        }
    }
}

/// A bitmap with a dense and a sparse case
#[derive(Debug, Clone, PartialEq, Eq, DagCbor)]
enum Bitmap {
    Dense(DenseBitmap),
    Sparse(SparseBitmap),
}

impl From<SparseBitmap> for Bitmap {
    fn from(value: SparseBitmap) -> Self {
        Self::Sparse(value)
    }
}

impl From<DenseBitmap> for Bitmap {
    fn from(value: DenseBitmap) -> Self {
        Self::Dense(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct DenseBitmap(Vec<IndexMask>);

impl DenseBitmap {
    fn rows(&self) -> BitmapRowsIter<'_> {
        BitmapRowsIter::Dense(self.0.iter())
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

impl From<DenseBitmap> for SparseBitmap {
    fn from(value: DenseBitmap) -> Self {
        Self(
            value
                .0
                .into_iter()
                .map(|mask| OneBitsIterator(mask).collect())
                .collect(),
        )
    }
}

impl<I: IntoIterator<Item = u32>> FromIterator<I> for Bitmap {
    fn from_iter<T: IntoIterator<Item = I>>(iter: T) -> Self {
        let mut res = Bitmap::default();
        for set in iter.into_iter() {
            res = res.push(set.into_iter())
        }
        res
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct SparseBitmap(Vec<IndexSet>);

impl SparseBitmap {
    fn rows(&self) -> BitmapRowsIter<'_> {
        BitmapRowsIter::Sparse(self.0.iter())
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

impl Default for Bitmap {
    fn default() -> Self {
        Bitmap::Dense(Default::default())
    }
}

impl Bitmap {
    fn rows(&self) -> BitmapRowsIter<'_> {
        match self {
            Self::Dense(x) => x.rows(),
            Self::Sparse(x) => x.rows(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Dense(x) => x.len(),
            Self::Sparse(x) => x.len(),
        }
    }

    fn push(self, iter: impl IntoIterator<Item = u32>) -> Self {
        match self {
            Self::Dense(mut inner) => match to_mask_or_set(iter) {
                Ok(mask) => {
                    inner.0.push(mask);
                    inner.into()
                }
                Err(set) => {
                    let mut inner = SparseBitmap::from(inner);
                    inner.0.push(set);
                    inner.into()
                }
            },
            Self::Sparse(mut inner) => {
                inner.0.push(iter.into_iter().collect());
                inner.into()
            }
        }
    }
}

impl Encode<DagCborCodec> for SparseBitmap {
    fn encode<W: std::io::Write>(&self, c: DagCborCodec, w: &mut W) -> anyhow::Result<()> {
        let mut rows: Vec<Vec<u32>> = self
            .rows()
            .map(|row_iter| row_iter.collect::<Vec<_>>())
            .collect();
        rows.iter_mut().for_each(|row| delta_encode(row));
        rows.encode(c, w)?;
        Ok(())
    }
}

impl Encode<DagCborCodec> for DenseBitmap {
    fn encode<W: std::io::Write>(&self, c: DagCborCodec, w: &mut W) -> anyhow::Result<()> {
        let mut rows: Vec<Vec<u32>> = self
            .rows()
            .map(|row_iter| row_iter.collect::<Vec<_>>())
            .collect();
        rows.iter_mut().for_each(|row| delta_encode(row));
        rows.encode(c, w)?;
        Ok(())
    }
}

impl Decode<DagCborCodec> for SparseBitmap {
    fn decode<R: std::io::Read + std::io::Seek>(
        c: DagCborCodec,
        r: &mut R,
    ) -> anyhow::Result<Self> {
        let mut rows: Vec<Vec<u32>> = Decode::decode(c, r)?;
        rows.iter_mut().for_each(|row| delta_decode(row));
        Ok(Self(
            rows.into_iter()
                .map(|row| row.into_iter().collect::<VecSet<_>>())
                .collect(),
        ))
    }
}

impl Decode<DagCborCodec> for DenseBitmap {
    fn decode<R: std::io::Read + std::io::Seek>(
        c: DagCborCodec,
        r: &mut R,
    ) -> anyhow::Result<Self> {
        let mut rows: Vec<Vec<u32>> = Decode::decode(c, r)?;
        rows.iter_mut().for_each(|row| delta_decode(row));
        Ok(Self(
            rows.into_iter()
                .map(|row| mask_from_bits_iter(row))
                .collect::<anyhow::Result<Vec<_>>>()?,
        ))
    }
}

enum BitmapRowsIter<'a> {
    Dense(slice::Iter<'a, IndexMask>),
    Sparse(slice::Iter<'a, IndexSet>),
}

impl<'a> Iterator for BitmapRowsIter<'a> {
    type Item = BitmapRowIter<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Dense(x) => x.next().map(|x| BitmapRowIter::Dense(OneBitsIterator(*x))),
            Self::Sparse(x) => x.next().map(|x| BitmapRowIter::Sparse(x.as_ref().iter())),
        }
    }
}

enum BitmapRowIter<'a> {
    Dense(OneBitsIterator),
    Sparse(slice::Iter<'a, u32>),
}

impl<'a> Iterator for BitmapRowIter<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Dense(x) => x.next(),
            Self::Sparse(x) => x.next().cloned(),
        }
    }
}

impl Display for DnfQuery {
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
pub struct TagSetSet {
    tags: FnvHashMap<Tag, u32>,
    sets: Bitmap,
}

#[derive(Debug, Clone)]
struct TagIndex {
    tags: TagSetSet,
    events: Vec<u32>,
}

impl TagIndex {
    pub fn from_elements(e: &[TagSet]) -> Self {
        let mut builder = TagSetSetBuilder::default();
        let events = e
            .iter()
            .map(|set| builder.push_ref(set).unwrap())
            .collect::<Vec<_>>();
        Self {
            tags: builder.tagsetset(),
            events,
        }
    }
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

impl From<DnfQuery> for TagSetSet {
    fn from(mut value: DnfQuery) -> Self {
        let tags: FnvHashMap<Tag, u32> = SliceIntoIter(value.tags.iter_mut())
            .enumerate()
            .map(|(index, tag)| (tag, index as u32))
            .collect();
        Self {
            tags,
            sets: value.sets,
        }
    }
}

impl From<TagSetSet> for DnfQuery {
    fn from(value: TagSetSet) -> Self {
        let mut tags = vec![None; value.tags.len()];
        for (tag, index) in value.tags {
            tags[index as usize] = Some(tag)
        }
        let tags = tags.into_iter().filter_map(|x| x).collect();
        Self {
            tags,
            sets: value.sets,
        }
    }
}

impl TagSetSet {
    pub fn dnf_query(&self, dnf: &DnfQuery, result: &mut [bool]) {
        let translate = dnf
            .tags
            .iter()
            .map(|tag| self.tags.get(tag).cloned())
            .collect::<Box<_>>();
        dnf_query0(&self.sets, dnf, &translate, result);
    }

    pub fn dnf_query_res(&self, dnf: &DnfQuery) -> Vec<bool> {
        let mut result = vec![true; self.sets.len()];
        self.dnf_query(dnf, &mut result);
        result
    }
}

impl DnfQuery {
    pub fn iter<'a>(&'a self) -> TagSetSetIter<'a> {
        TagSetSetIter(&self.tags, self.sets.rows())
    }

    pub fn dnf_query(&self, dnf: &DnfQuery) -> Vec<bool> {
        let lookup = self.index_lut();
        let translate = dnf
            .tags
            .iter()
            .map(|tag| lookup.get(tag).cloned())
            .collect::<Box<_>>();
        let mut result = vec![true; self.sets.len()];
        dnf_query0(&self.sets, dnf, &translate, &mut result);
        result
    }

    fn index_lut(&self) -> FnvHashMap<&Tag, u32> {
        self.tags
            .iter()
            .enumerate()
            .map(|(index, tag)| (tag, u32::try_from(index).unwrap()))
            .collect()
    }
}

/// performs a dnf query on an index, given a lookup table to translate from the dnf query to the index domain
fn dnf_query0(index: &Bitmap, dnf: &DnfQuery, translate: &[Option<u32>], result: &mut [bool]) {
    match index {
        Bitmap::Sparse(index) => {
            let dnf = SparseBitmap(
                dnf.sets
                    .rows()
                    .filter_map(|row| {
                        row.map(|index| translate[index as usize])
                            .collect::<Option<IndexSet>>()
                    })
                    .collect(),
            );
            for i in 0..index.len() {
                result[i] = result[i] && {
                    let set = &index.0[i];
                    dnf.0.iter().any(move |query| query.is_subset(set))
                }
            }
        }
        Bitmap::Dense(index) => {
            let dnf = DenseBitmap(
                dnf.sets
                    .rows()
                    .filter_map(|row| {
                        mask_from_bits_iter(
                            row.map(|index| translate[index as usize].unwrap_or(128)),
                        )
                        .ok()
                    })
                    .collect(),
            );
            for i in 0..index.len() {
                result[i] = result[i] && {
                    let set = &index.0[i];
                    dnf.0.iter().any(move |query| is_subset(*query, *set))
                }
            }
        }
    }
}

pub struct TagSetSetIter<'a>(&'a [Tag], BitmapRowsIter<'a>);

impl<'a> Iterator for TagSetSetIter<'a> {
    type Item = TagRefIter<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.1.next().map(|iter| TagRefIter(self.0, iter))
    }
}

pub struct TagRefIter<'a>(&'a [Tag], BitmapRowIter<'a>);

impl<'a> Iterator for TagRefIter<'a> {
    type Item = &'a Tag;

    fn next(&mut self) -> Option<Self::Item> {
        self.1.next().map(|index| &self.0[index as usize])
    }
}

#[derive(Debug, Default)]
pub struct TagSetSetBuilder {
    tags: FnvHashMap<Tag, u32>,
    sets: FnvHashMap<IndexSet, u32>,
}

impl TagSetSetBuilder {
    pub fn tagsetset(self) -> TagSetSet {
        let tags = self.tags;
        let mut sets = vec![IndexSet::default(); self.sets.len()];
        for (set, index) in self.sets {
            sets[index as usize] = set
        }
        let sets = sets.into_iter().map(|x| x.into_iter()).collect();
        TagSetSet { tags, sets }
    }

    pub fn result(self) -> DnfQuery {
        let mut tags = vec![None; self.tags.len()];
        for (tag, index) in self.tags {
            tags[index as usize] = Some(tag)
        }
        let tags = tags.into_iter().filter_map(|x| x).collect();
        let mut sets = vec![IndexSet::default(); self.sets.len()];
        for (set, index) in self.sets {
            sets[index as usize] = set
        }
        let sets = sets.into_iter().map(|x| x.into_iter()).collect();
        DnfQuery { tags, sets }
    }

    pub fn push_ref(&mut self, tags: &TagSet) -> anyhow::Result<u32> {
        let indices = tags.iter().map(|tag| self.add_tag_ref(tag));
        let set = indices.collect::<anyhow::Result<IndexSet>>()?;
        let offset = u32::try_from(self.sets.len())?;
        self.sets.insert(set, offset);
        Ok(offset)
    }

    fn add_tag_ref(&mut self, tag: &Tag) -> anyhow::Result<u32> {
        Ok(if let Some(index) = self.tags.get(tag) {
            *index
        } else {
            let index = u32::try_from(self.tags.len())?;
            self.tags.insert(tag.clone(), index);
            index
        })
    }
}

fn delta_encode<T: WrappingSub<Output = T> + Copy>(data: &mut [T]) {
    for i in (1..data.len()).rev() {
        data[i] = data[i].wrapping_sub(&data[i - 1]);
    }
}

fn delta_decode<T: WrappingAdd<Output = T> + Copy>(data: &mut [T]) {
    for i in 1..data.len() {
        data[i] = data[i].wrapping_add(&data[i - 1]);
    }
}

fn is_subset(a: IndexMask, b: IndexMask) -> bool {
    a & b == a
}

pub struct OneBitsIterator(IndexMask);

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
fn mask_from_bits_iter(iterator: impl IntoIterator<Item = u32>) -> anyhow::Result<IndexMask> {
    let mut mask: IndexMask = 0;
    let mut iter = iterator.into_iter();
    while let Some(bit) = iter.next() {
        anyhow::ensure!(bit < 128);
        mask |= 1u128 << bit;
    }
    Ok(mask)
}

/// Given an iterator of bits, creates either a 128 bit bitmask, or a set of bits.
fn to_mask_or_set(iterator: impl IntoIterator<Item = u32>) -> result::Result<IndexMask, IndexSet> {
    let mut mask: IndexMask = 0;
    let mut iter = iterator.into_iter();
    while let Some(bit) = iter.next() {
        if bit < 128 {
            mask |= 1u128 << bit;
        } else {
            let mut res = OneBitsIterator(mask).collect::<FnvHashSet<_>>();
            res.insert(bit);
            res.extend(iter);
            return Err(res.into_iter().collect());
        }
    }
    Ok(mask)
}

fn tag_set(tags: &[&str]) -> TagSet {
    tags.iter()
        .map(|x| Tag(x.to_string().into_bytes().into()))
        .collect()
}

fn main() -> anyhow::Result<()> {
    let a = tag_set(&["a", "b", "x", "y"]);
    let b = tag_set(&["a", "x", "y"]);
    let c = tag_set(&["a", "b", "y"]);
    let mut builder = TagSetSetBuilder::default();
    builder.push_ref(&a)?;
    builder.push_ref(&b)?;
    builder.push_ref(&c)?;
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
    use libipld::codec::Codec;
    use quickcheck::Arbitrary;

    fn tag_set_set(tags: &[&[&str]]) -> DnfQuery {
        let mut builder = TagSetSetBuilder::default();
        for tags in tags.iter().map(|x| tag_set(x)) {
            builder.push_ref(&tags).unwrap();
        }
        builder.result()
    }

    use super::*;

    #[test]
    fn dnf_query() {
        {
            let tags = tag_set_set(&[&["a", "b"], &["a", "c"], &["b", "c"]]);
            let dnf = tag_set_set(&[&["a"]]);
            let indexes = tags.dnf_query(&dnf);
            assert_eq!(&indexes, &[true, true, false]);
        }
        {
            let tags = tag_set_set(&[&["a", "b"], &["a", "c"], &["b", "c"]]);
            let dnf = tag_set_set(&[&["a", "b"]]);
            let indexes = tags.dnf_query(&dnf);
            assert_eq!(&indexes, &[true, false, false]);
        }
        {
            let tags = tag_set_set(&[&["a", "b"], &["a", "c"], &["b", "c"]]);
            let dnf = tag_set_set(&[&["a", "x"]]);
            let indexes = tags.dnf_query(&dnf);
            assert_eq!(&indexes, &[false, false, false]);
        }
        {
            let tags = tag_set_set(&[&["a", "b"], &["a", "c"], &["b", "c"]]);
            let dnf = tag_set_set(&[&[]]);
            let indexes = tags.dnf_query(&dnf);
            assert_eq!(&indexes, &[true, true, true]);
        }
        {
            let tags = tag_set_set(&[&["a", "b"], &["a", "c"], &["b", "c"]]);
            let dnf = tag_set_set(&[&["c"]]);
            let indexes = tags.dnf_query(&dnf);
            assert_eq!(&indexes, &[false, true, true]);
        }
        {
            let tags = tag_set_set(&[&["a", "b"], &["b", "c"], &["c", "d"]]);
            let dnf = tag_set_set(&[&["a"], &["d"]]);
            let indexes = tags.dnf_query(&dnf);
            assert_eq!(&indexes, &[true, false, true]);
        }
    }

    #[quickcheck]
    fn bits_iter_roundtrip(value: IndexMask) -> bool {
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
    fn tag_set_set_cbor_roundtrip(value: DnfQuery) -> bool {
        let bytes = DagCborCodec.encode(&value).unwrap();
        let value1 = DagCborCodec.decode(&bytes).unwrap();
        value == value1
    }

    impl Arbitrary for Tag {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let tag = g.choose(TAG_NAMES).unwrap();
            Tag(tag.as_bytes().to_vec().into())
        }
    }

    impl Arbitrary for DnfQuery {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let tags: FnvHashSet<Tag> = Arbitrary::arbitrary(g);
            let mut sets: Vec<IndexMask> = Arbitrary::arbitrary(g);
            let mut tags = tags.into_iter().collect::<Vec<_>>();
            tags.truncate(128);
            let mask = !(IndexMask::max_value() << tags.len());
            sets.iter_mut().for_each(|set| *set &= mask);
            sets.sort();
            sets.dedup();
            Self {
                sets: DenseBitmap(sets.into()).into(),
                tags: tags.into(),
            }
        }
    }

    const TAG_NAMES: &[&'static str] = &["a", "b", "c", "d", "e", "f"];
}
