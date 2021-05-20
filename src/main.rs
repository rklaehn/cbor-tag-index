use fmt::{Display, Write};
use fnv::{FnvHashMap, FnvHashSet};
use libipld::DagCbor;
use std::{convert::TryFrom, fmt, iter::FromIterator, mem::swap};
mod bitmap;
use bitmap::*;

#[cfg(test)]
extern crate quickcheck;
#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

/// our toy tag
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, DagCbor)]
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
#[derive(Debug, Clone, Default)]
pub struct TagSetSet {
    tags: FnvHashMap<Tag, u32>,
    sets: Bitmap,
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
#[derive(Debug, Clone)]
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

    pub fn len(&self) -> usize {
        self.events.len()
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
        let mut result = vec![true; self.sets.rows()];
        self.dnf_query(dnf, &mut result);
        result
    }
}

impl DnfQuery {
    pub fn iter<'a>(&'a self) -> TagSetSetIter<'a> {
        TagSetSetIter(&self.tags, self.sets.iter())
    }

    pub fn dnf_query(&self, dnf: &DnfQuery) -> Vec<bool> {
        let lookup = self.index_lut();
        let translate = dnf
            .tags
            .iter()
            .map(|tag| lookup.get(tag).cloned())
            .collect::<Box<_>>();
        let mut result = vec![true; self.sets.rows()];
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

    // create a sequence of tag sets
    fn tss(tags: &str) -> Vec<TagSet> {
        let parts = tags.split(",");
        parts.map(ts).collect()
    }

    fn tag_set_set(tags: &[&[&str]]) -> DnfQuery {
        let mut builder = TagSetSetBuilder::default();
        for tags in tags.iter().map(|x| tag_set(x)) {
            builder.push(&tags).unwrap();
        }
        builder.dnf_query()
    }

    use super::*;

    #[test]
    fn tag_index_from_elements() -> anyhow::Result<()> {
        let index = TagIndex::new(&[
            tag_set(&["a"]),
            tag_set(&["a", "b"]),
            tag_set(&["b", "c"]),
            tag_set(&["a"]),
        ])?;
        let query = DnfQuery::new(&[])?;
        let mut mask = [true; 4];
        query.set_matching(&index, &mut mask);
        println!("{:?}", query);
        println!("{:?}", index);
        println!("{:?}", mask);
        Ok(())
    }

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
                sets: DenseBitmap::new(sets.into()).into(),
                tags: tags.into(),
            }
        }
    }

    const TAG_NAMES: &[&'static str] = &["a", "b", "c", "d", "e", "f"];
}
