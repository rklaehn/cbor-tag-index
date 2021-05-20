use fmt::{Display, Write};
use fnv::{FnvHashMap, FnvHashSet};
use libipld::{
    codec::{Decode, Encode},
    DagCbor,
};
use libipld_cbor::DagCborCodec;
use std::{convert::TryFrom, fmt, iter::FromIterator, mem::swap, ops::Index, usize};
mod bitmap;
mod util;
use bitmap::*;

#[cfg(test)]
extern crate quickcheck;
#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

/// our toy tag
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, DagCbor)]
pub struct Tag(Box<[u8]>);

/// a set of tags
type TagSet = FnvHashSet<Tag>;

/// A compact representation of a seq of tag sets,
///
/// to be used as a DNF query.
///
/// `tags` are a sequence of strings, where the offset corresponds to the
/// set bit in the bitmap.
///
/// E.g. ("a" & "b") | ("b" & "c") | ("d") would be encoded as
///
/// {
///   tags: ["a", "b", "c", "d"],
///   sets: [
///     b0011,
///     b0110,
///     b1000,
///   ]
/// }
#[derive(Debug, Clone, Default, PartialEq, Eq, DagCbor)]
pub struct DnfQuery {
    tags: Vec<Tag>,
    sets: Bitmap,
}

/// Same as a [DnfQuery], but with optimized support for translation.
///
/// In this representation, `tags` is a map from tag to index.
///
/// E.g. ("a" & "b") | ("b" & "c") | ("d") would be encoded as
///
/// {
///   tags: {
///     "a": 0,
///     "b": 1,
///     "c": 2,
///     "d": 3
///   },
///   sets: [
///     b0011,
///     b0110,
///     b1000,
///   ]
/// }
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TagSetSet {
    tags: FnvHashMap<Tag, u32>,
    sets: Bitmap,
}

impl Encode<DagCborCodec> for TagSetSet {
    fn encode<W: std::io::Write>(&self, c: DagCborCodec, w: &mut W) -> anyhow::Result<()> {
        DnfQuery::from(self.clone()).encode(c, w)
    }
}

impl Decode<DagCborCodec> for TagSetSet {
    fn decode<R: std::io::Read + std::io::Seek>(
        c: DagCborCodec,
        r: &mut R,
    ) -> anyhow::Result<Self> {
        DnfQuery::decode(c, r).map(Into::into)
    }
}

/// A tag index, using a [TagSetSet] to encode the distinct tag sets, and a vector
/// of offsets for each event.
///
/// A sequence of events with the following tag sets:
///
///```
/// [{"a"}, {"a", "b"}, {"b","c"}, {"a"}]
///```
///
/// would be encoded like this:
///
///```
/// {
///   tags: {
///     tags: { "a": 0, "b": 1, "c": 2 },
///     sets: [
///       b001, //   a
///       b010, //  b
///       b110, // cb
///   },
///   events: [
///     0, // first bitset
///     1, // 2nd bitset
///     2, // 3rd bitset
///     0, // first bitset again
///   ],
/// }
///```
#[derive(Debug, Clone, PartialEq, Eq, DagCbor)]
pub struct TagIndex {
    /// efficiently encoded distinct tags
    tags: TagSetSet,
    /// tag offset for each event
    events: Vec<u32>,
}

impl fmt::Display for Tag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Ok(text) = std::str::from_utf8(&self.0) {
            write!(f, "{}", text)
        } else {
            write!(f, "{}", hex::encode(&self.0))
        }
    }
}

impl fmt::Debug for Tag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Ok(text) = std::str::from_utf8(&self.0) {
            write!(f, "{}", text)
        } else {
            write!(f, "{}", hex::encode(&self.0))
        }
    }
}

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

impl DnfQuery {
    pub fn new<'a>(sets: impl IntoIterator<Item = &'a TagSet>) -> anyhow::Result<Self> {
        let mut builder = TagSetSetBuilder::default();
        for set in sets {
            builder.push(&set)?;
        }
        Ok(builder.dnf_query())
    }

    /// given a bitmap of matches, corresponding to the events in the index,
    /// set those bytes to false that do not match.
    pub fn set_matching(&self, index: &TagIndex, matches: &mut [bool]) {
        // create a bool array corresponding to the distinct tagsets in the index
        let mut tmp = vec![false; index.tags.sets.rows()];
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

impl TagIndex {
    pub fn new(e: &[TagSet]) -> anyhow::Result<Self> {
        let mut builder = TagSetSetBuilder::default();
        let events = e
            .iter()
            .map(|set| builder.push(set))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self {
            tags: builder.tag_set_set(),
            events,
        })
    }

    pub fn tags(&self) -> impl Iterator<Item = TagSet> + '_ {
        let lut = self.tags.tags().collect::<Vec<_>>();
        self.events
            .iter()
            .map(move |offset| lut[*offset as usize].clone())
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }
}

impl Display for TagIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(
                self.tags()
                    .map(|ts| ts.into_iter().collect::<DebugUsingDisplay<_>>()),
            )
            .finish()
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
            .map(|(index, tag)| (tag, u32::try_from(index).unwrap()))
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
        let tags: Vec<Tag> = tags.into_iter().filter_map(|x| x).collect();
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

    fn lut(&self) -> Vec<Tag> {
        let mut tags = vec![None; self.tags.len()];
        for (tag, index) in &self.tags {
            tags[*index as usize] = Some(tag)
        }
        tags.into_iter().filter_map(|x| x.cloned()).collect()
    }

    /// get back the tag sets the tag index was created from, in order
    pub fn tags(&self) -> impl Iterator<Item = TagSet> + '_ {
        let lut = self.lut();
        self.sets
            .iter()
            .map(move |row| row.map(|i| lut[i as usize].clone()).collect::<TagSet>())
    }
}

impl DnfQuery {
    pub fn iter<'a>(&'a self) -> TagSetSetIter<'a> {
        TagSetSetIter(&self.tags, self.sets.iter())
    }

    /// get back the tag sets the dnf query
    pub fn tags(&self) -> impl Iterator<Item = TagSet> + '_ {
        self.sets.iter().map(move |rows| {
            rows.map(|index| self.tags[index as usize].clone())
                .collect()
        })
    }
}

/// performs a dnf query on an index, given a lookup table to translate from the dnf query to the index domain
fn dnf_query0(index: &Bitmap, dnf: &DnfQuery, translate: &[Option<u32>], result: &mut [bool]) {
    match index {
        Bitmap::Sparse(index) => {
            let dnf = SparseBitmap::new(
                dnf.sets
                    .iter()
                    .filter_map(|row| {
                        row.map(|index| translate[index as usize])
                            .collect::<Option<IndexSet>>()
                    })
                    .collect(),
            );
            for row in 0..index.rows() {
                result[row] = result[row] && {
                    let set = &index[row];
                    dnf.iter().any(move |query| query.is_subset(set))
                }
            }
        }
        Bitmap::Dense(index) => {
            let dnf = DenseBitmap::new(
                dnf.sets
                    .iter()
                    .filter_map(|row| {
                        mask_from_bits_iter(
                            row.map(|index| translate[index as usize].unwrap_or(128)),
                        )
                        .ok()
                    })
                    .collect(),
            );
            for i in 0..index.rows() {
                result[i] = result[i] && {
                    let set = &index[i];
                    dnf.iter().any(move |query| is_subset(*query, *set))
                }
            }
        }
    }
}

fn is_subset(a: IndexMask, b: IndexMask) -> bool {
    a & b == a
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
    /// Return the result as a [TagSetSet]
    pub fn tag_set_set(self) -> TagSetSet {
        let tags = self.tags;
        let mut sets = vec![IndexSet::default(); self.sets.len()];
        for (set, index) in self.sets {
            sets[index as usize] = set
        }
        let sets = sets.into_iter().map(|x| x.into_iter()).collect();
        TagSetSet { tags, sets }
    }

    /// Return the result as a [DnfQuery]
    pub fn dnf_query(self) -> DnfQuery {
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

    pub fn push(&mut self, tags: &TagSet) -> anyhow::Result<u32> {
        let indices = tags.iter().map(|tag| self.add_tag(tag));
        let set = indices.collect::<anyhow::Result<IndexSet>>()?;
        Ok(if let Some(index) = self.sets.get(&set) {
            *index
        } else {
            let index = u32::try_from(self.sets.len())?;
            self.sets.insert(set, index);
            index
        })
    }

    fn add_tag(&mut self, tag: &Tag) -> anyhow::Result<u32> {
        Ok(if let Some(index) = self.tags.get(tag) {
            *index
        } else {
            let index = u32::try_from(self.tags.len())?;
            self.tags.insert(tag.clone(), index);
            index
        })
    }
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
    builder.push(&a)?;
    builder.push(&b)?;
    builder.push(&c)?;
    let t = builder.dnf_query();
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
    use libipld_cbor::DagCborCodec;
    use quickcheck::Arbitrary;

    // create a test tag set - each alphanumeric char will be converted to an individual tag.
    fn ts(tags: &str) -> TagSet {
        tags.chars()
            .filter(|c| char::is_alphanumeric(*c))
            .map(|x| Tag(x.to_string().into_bytes().into()))
            .collect()
    }

    // create a sequence of tag sets, separated by ,
    fn tss(tags: &str) -> Vec<TagSet> {
        tags.split(",").map(ts).collect()
    }

    // create a dnf query, separated by |
    fn dnf(tags: &str) -> DnfQuery {
        let parts = tags.split("|").map(ts).collect::<Vec<_>>();
        DnfQuery::new(&parts).unwrap()
    }

    // create a dnf query, separated by |
    fn ti(tags: &str) -> TagIndex {
        TagIndex::new(&tss(tags)).unwrap()
    }

    use super::*;

    fn matches(index: &str, query: &str) -> String {
        let index = ti(index);
        let query = dnf(query);
        let mut matches = vec![true; index.len()];
        query.set_matching(&index, &mut matches);
        matches.iter().map(|x| if *x { '1' } else { '0' }).collect()
    }

    #[test]
    fn tag_index_query_tests() -> anyhow::Result<()> {
        assert_eq!(&matches(" a,ab,bc, a", "ab"), "0100");
        assert_eq!(&matches(" a, a, a, a", "ab"), "0000");
        assert_eq!(&matches(" a, a, a,ab", "ab|c|d"), "0001");
        Ok(())
    }

    #[test]
    fn dnf_query() {
        // sequence containing empty sets, matches everything
        assert_eq!(&matches("ab,ac,bc", " ||"), "111");
        assert_eq!(&matches("ab,ac,bc", "  a"), "110");
        assert_eq!(&matches("ab,ac,bc", " ab"), "100");
        assert_eq!(&matches("ab,ac,bc", " ax"), "000");
        assert_eq!(&matches("ab,ac,bc", "  c"), "011");
        assert_eq!(&matches("ab,bc,cd", "a|d"), "101");
    }

    #[quickcheck]
    fn dnf_query_cbor_roundtrip(value: DnfQuery) -> bool {
        let bytes = DagCborCodec.encode(&value).unwrap();
        let value1 = DagCborCodec.decode(&bytes).unwrap();
        value == value1
    }

    #[quickcheck]
    fn tag_set_set_cbor_roundtrip(value: TagSetSet) -> bool {
        let bytes = DagCborCodec.encode(&value).unwrap();
        let value1 = DagCborCodec.decode(&bytes).unwrap();
        value == value1
    }

    #[quickcheck]
    fn tag_index_cbor_roundtrip(value: TagIndex) -> bool {
        let bytes = DagCborCodec.encode(&value).unwrap();
        let value1 = DagCborCodec.decode(&bytes).unwrap();
        value == value1
    }

    #[quickcheck]
    fn set_matching(index: TagIndex, query: DnfQuery) -> bool {
        let mut bits1 = vec![true; index.len()];
        let mut bits2 = vec![false; index.len()];
        query.set_matching(&index, &mut bits1);

        let query_tags = query.tags().collect::<Vec<_>>();
        for (tags, matching) in index.tags().zip(bits2.iter_mut()) {
            *matching = query_tags.iter().any(|q| q.is_subset(&tags))
        }
        let bt = bits1
            .iter()
            .map(|x| if *x { '1' } else { '0' })
            .collect::<String>();
        println!("{} {} {}", index, query, bt);
        bits1 == bits2
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
                sets: DenseBitmap::new(sets.into()).into(),
                tags: tags.into(),
            }
        }
    }

    impl Arbitrary for TagSetSet {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let q: DnfQuery = Arbitrary::arbitrary(g);
            q.into()
        }
    }

    impl Arbitrary for TagIndex {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let tags: Vec<TagSet> = Arbitrary::arbitrary(g);
            TagIndex::new(&tags).unwrap()
        }
    }

    const TAG_NAMES: &[&'static str] = &["a", "b", "c", "d", "e", "f"];
}
