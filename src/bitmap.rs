use super::util::{read_seq, IterExt};
use core::slice;
use fnv::FnvHashSet;
use libipld::{
    cbor::DagCborCodec,
    codec::{Decode, Encode},
};
use std::{io, iter::FromIterator, ops::Index, result, usize};
use vec_collections::VecSet;

// set for the sparse case
pub(crate) type IndexSet = VecSet<[u32; 4]>;
// mask for the dense case
pub(crate) type IndexMask = u128;

/// A bitmap with a dense and a sparse case
#[derive(Debug, Clone, PartialEq, Eq)]
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
    pub fn new(items: impl IntoIterator<Item = impl IntoIterator<Item = u32>>) -> Self {
        Self::from_iter(items.into_iter())
    }

    pub fn is_dense(&self) -> bool {
        match self {
            Self::Dense(_) => true,
            Self::Sparse(_) => false,
        }
    }

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

    pub fn push_row(self, row: BitmapRow) -> Self {
        match self {
            Self::Dense(mut inner) => match row.0 {
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
                inner.0.push(row.into_sparse());
                inner.into()
            }
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

impl Encode<DagCborCodec> for Bitmap {
    fn encode<W: io::Write>(&self, c: DagCborCodec, w: &mut W) -> anyhow::Result<()> {
        match self {
            Self::Dense(x) => x.encode(c, w),
            Self::Sparse(x) => x.encode(c, w),
        }
    }
}

impl Encode<DagCborCodec> for SparseBitmap {
    fn encode<W: io::Write>(&self, c: DagCborCodec, w: &mut W) -> anyhow::Result<()> {
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
    fn encode<W: io::Write>(&self, c: DagCborCodec, w: &mut W) -> anyhow::Result<()> {
        let mut rows: Vec<Vec<u32>> = self
            .iter()
            .map(|row| OneBitsIterator(*row).collect::<Vec<_>>())
            .collect();
        rows.iter_mut().for_each(|row| delta_encode(row));
        rows.encode(c, w)?;
        Ok(())
    }
}

impl FromIterator<BitmapRow> for Bitmap {
    fn from_iter<T: IntoIterator<Item = BitmapRow>>(iter: T) -> Self {
        let mut res = Bitmap::default();
        for row in iter.into_iter() {
            res = res.push_row(row)
        }
        res
    }
}

impl Decode<DagCborCodec> for Bitmap {
    fn decode<R: io::Read + io::Seek>(_: DagCborCodec, r: &mut R) -> anyhow::Result<Self> {
        read_seq::<_, R, BitmapRow>(r)
    }
}

impl Decode<DagCborCodec> for SparseBitmap {
    fn decode<R: io::Read + io::Seek>(c: DagCborCodec, r: &mut R) -> anyhow::Result<Self> {
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
    fn decode<R: io::Read + io::Seek>(c: DagCborCodec, r: &mut R) -> anyhow::Result<Self> {
        let mut rows: Vec<Vec<u32>> = Decode::decode(c, r)?;
        rows.iter_mut().for_each(|row| delta_decode(row));
        Ok(Self(
            rows.into_iter()
                .map(mask_from_bits_iter)
                .collect::<anyhow::Result<Vec<_>>>()?,
        ))
    }
}

pub(crate) enum BitmapRowsIter<'a> {
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

fn delta_encode(data: &mut [u32]) {
    for i in (1..data.len()).rev() {
        data[i] = data[i].wrapping_sub(data[i - 1]);
    }
}

fn delta_decode(data: &mut [u32]) {
    for i in 1..data.len() {
        data[i] = data[i].wrapping_add(data[i - 1]);
    }
}

pub(crate) struct OneBitsIterator(IndexMask);

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
    let iter = iterator.into_iter();
    for bit in iter {
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

/// A single row of a dense or sparse bitmap
pub(crate) struct BitmapRow(
    /// ok -> result fits into a IndexMask
    /// err -> we had to use an IndexSet
    std::result::Result<IndexMask, IndexSet>,
);

impl BitmapRow {
    pub fn into_sparse(self) -> IndexSet {
        match self.0 {
            Ok(mask) => OneBitsIterator(mask).collect(),
            Err(set) => set,
        }
    }
}

impl FromIterator<u32> for BitmapRow {
    fn from_iter<I: IntoIterator<Item = u32>>(iter: I) -> Self {
        // add the delta decoding
        let mut offset = 0u32;
        let iter = iter.into_iter().map(move |x| {
            offset = offset.wrapping_add(x);
            offset
        });
        BitmapRow(to_mask_or_set(iter))
    }
}

impl Decode<DagCborCodec> for BitmapRow {
    fn decode<R: io::Read + io::Seek>(_: DagCborCodec, r: &mut R) -> anyhow::Result<Self> {
        read_seq::<_, R, u32>(r)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use libipld::{
        codec::{assert_roundtrip, Codec},
        ipld,
    };

    #[test]
    fn dense_1() {
        let bitmap = Bitmap::new(vec![vec![1, 2, 4, 8, 127]; 7]);
        assert!(bitmap.is_dense());
        assert_eq!(bitmap.rows(), 7);
        for i in 0..bitmap.rows() {
            assert_eq!(bitmap.row(i).collect::<Vec<_>>(), vec![1, 2, 4, 8, 127]);
        }
    }

    #[test]
    fn sparse_1() {
        let bitmap = Bitmap::new(vec![vec![1, 2, 4, 8, 128]; 9]);
        assert!(!bitmap.is_dense());
        assert_eq!(bitmap.rows(), 9);
        for i in 0..bitmap.rows() {
            assert_eq!(bitmap.row(i).collect::<Vec<_>>(), vec![1, 2, 4, 8, 128]);
        }
    }

    #[test]
    fn dense_ipld() {
        let bitmap = Bitmap::new(vec![vec![0, 127]]);
        assert!(bitmap.is_dense());
        let expected = ipld! {
            [[0, 127]]
        };
        assert_roundtrip(DagCborCodec, &bitmap, &expected);
    }

    #[test]
    fn sparse_ipld() {
        let bitmap = Bitmap::new(vec![vec![0, 128]]);
        assert!(!bitmap.is_dense());
        let expected = ipld! {
            [[0, 128]]
        };
        assert_roundtrip(DagCborCodec, &bitmap, &expected);
    }

    #[quickcheck]
    fn bitmap_roundtrip(value: Vec<HashSet<u32>>) -> bool {
        let bitmap = Bitmap::new(value.clone());
        let value1: Vec<HashSet<u32>> = bitmap.iter().map(|row| row.collect()).collect();
        value == value1
    }

    #[quickcheck]
    fn bits_iter_roundtrip(value: IndexMask) -> bool {
        let iter = OneBitsIterator(value);
        let value1 = mask_from_bits_iter(iter).unwrap();
        value == value1
    }

    #[quickcheck]
    fn delta_decode_roundtrip(mut values: Vec<u32>) -> bool {
        values.sort_unstable();
        values.dedup();
        let reference = values.clone();
        delta_encode(&mut values);
        delta_decode(&mut values);
        values == reference
    }

    #[quickcheck]
    fn dnf_query_cbor_roundtrip(value: Bitmap) -> bool {
        let bytes = DagCborCodec.encode(&value).unwrap();
        let value1 = DagCborCodec.decode(&bytes).unwrap();
        value == value1
    }
}
