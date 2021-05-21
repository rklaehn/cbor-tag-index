use std::collections::HashSet;
use quickcheck::Arbitrary;
use super::*;

impl Arbitrary for Bitmap {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let items: Vec<HashSet<u32>> = Arbitrary::arbitrary(g);
        Bitmap::new(items)
    }
}

impl Arbitrary for Tag {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let tag = g.choose(TAG_NAMES).unwrap();
        Tag(tag.as_bytes().to_vec().into())
    }
}

impl Arbitrary for DnfQuery {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let mut tags: Vec<TagSet> = Arbitrary::arbitrary(g);
        tags.truncate(3);
        DnfQuery::new(&tags).unwrap()
    }
}

impl Arbitrary for TagSetSet {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let mut tags: Vec<TagSet> = Arbitrary::arbitrary(g);
        tags.truncate(3);
        TagSetSet::new(&tags).unwrap()
    }
}

impl Arbitrary for TagIndex {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let tags: Vec<TagSet> = Arbitrary::arbitrary(g);
        TagIndex::new(&tags).unwrap()
    }
}

const TAG_NAMES: &[&'static str] = &["a", "b", "c", "d", "e", "f"];
