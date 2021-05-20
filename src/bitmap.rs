use core::slice;
use fnv::FnvHashSet;
use libipld::{
    cbor::DagCborCodec,
    codec::{Decode, Encode},
    DagCbor,
};
use num_traits::{WrappingAdd, WrappingSub};
use std::{iter::FromIterator, ops::Index, result, usize};
use vec_collections::VecSet;

use crate::util::IterExt;

// set for the sparse case
pub(crate) type IndexSet = VecSet<[u32; 4]>;
// mask for the dense case
pub(crate) type IndexMask = u128;

/// A bitmap with a dense and a sparse case
#[derive(Debug, Clone, PartialEq, Eq, DagCbor)]
pub(crate) enum Bitmap {
    Dense(DenseBitmap),
    Sparse(SparseBitmap),
}

impl Default for Bitmap {
    fn default() -> Self {
        Bitmap::Dense(Default::default())
    }
}

impl Bitmap {
    pub fn rows(&self) -> usize {
        match self {
            Self::Dense(x) => x.rows(),
            Self::Sparse(x) => x.rows(),
        }
    }

    pub fn row(&self, index: usize) -> impl Iterator<Item = u32> + '_ {
        match self {
            Self::Dense(x) => x.row(index).left_iter(),
            Self::Sparse(x) => x.row(index).right_iter(),
        }
    }

    pub fn push(self, iter: impl IntoIterator<Item = u32>) -> Self {
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

    pub fn iter(&self) -> BitmapRowsIter<'_> {
        match self {
            Bitmap::Sparse(x) => BitmapRowsIter::Sparse(x.iter()),
            Bitmap::Dense(x) => BitmapRowsIter::Dense(x.iter()),
        }
    }
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
pub(crate) struct DenseBitmap(Vec<IndexMask>);

impl DenseBitmap {
    pub fn new(rows: Vec<IndexMask>) -> Self {
        Self(rows)
    }

    pub fn rows(&self) -> usize {
        self.0.len()
    }

    pub fn row(&self, index: usize) -> impl Iterator<Item = u32> + '_ {
        OneBitsIterator(self.0[index])
    }

    pub fn iter(&self) -> std::slice::Iter<'_, IndexMask> {
        self.0.iter()
    }
}

impl Index<usize> for DenseBitmap {
    type Output = IndexMask;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
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
pub(crate) struct SparseBitmap(Vec<IndexSet>);

impl Index<usize> for SparseBitmap {
    type Output = IndexSet;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl SparseBitmap {
    pub fn new(rows: Vec<IndexSet>) -> Self {
        Self(rows)
    }

    pub fn rows(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, IndexSet> {
        self.0.iter()
    }

    pub fn row(&self, index: usize) -> impl Iterator<Item = u32> + '_ {
        self.0[index].iter().cloned()
    }
}

impl Encode<DagCborCodec> for SparseBitmap {
    fn encode<W: std::io::Write>(&self, c: DagCborCodec, w: &mut W) -> anyhow::Result<()> {
        let mut rows: Vec<Vec<u32>> = self
            .iter()
            .map(|row_iter| row_iter.into_iter().cloned().collect::<Vec<_>>())
            .collect();
        rows.iter_mut().for_each(|row| delta_encode(row));
        rows.encode(c, w)?;
        Ok(())
    }
}

impl Encode<DagCborCodec> for DenseBitmap {
    fn encode<W: std::io::Write>(&self, c: DagCborCodec, w: &mut W) -> anyhow::Result<()> {
        let mut rows: Vec<Vec<u32>> = self
            .iter()
            .map(|row| OneBitsIterator(*row).collect::<Vec<_>>())
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

pub(crate) enum BitmapRowsIter<'a> {
    Dense(slice::Iter<'a, IndexMask>),
    Sparse(slice::Iter<'a, IndexSet>),
}

pub(crate) enum BitmapRowsIntoIter {
    Dense(u64),
    Sparse(u64),
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

pub(crate) enum BitmapRowIter<'a> {
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
pub fn mask_from_bits_iter(iterator: impl IntoIterator<Item = u32>) -> anyhow::Result<IndexMask> {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
