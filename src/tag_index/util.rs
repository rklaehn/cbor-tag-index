/// Like the one from itertools, but more convenient
pub(crate) enum EitherIter<L, R> {
    Left(L),
    Right(R),
}

impl<L, R, T> Iterator for EitherIter<L, R>
where
    L: Iterator<Item = T>,
    R: Iterator<Item = T>,
{
    type Item = T;
    fn next(&mut self) -> std::option::Option<<Self as Iterator>::Item> {
        match self {
            Self::Left(l) => l.next(),
            Self::Right(r) => r.next(),
        }
    }
}

#[allow(dead_code)]
pub(crate) type BoxedIter<'a, T> = Box<dyn Iterator<Item = T> + Send + 'a>;

/// Some convenience fns so we don't have to depend on IterTools
pub(crate) trait IterExt<'a>
where
    Self: Iterator + Sized + Send + 'a,
{
    fn boxed(self) -> BoxedIter<'a, Self::Item> {
        Box::new(self)
    }

    fn left_iter<R>(self) -> EitherIter<Self, R> {
        EitherIter::Left(self)
    }

    fn right_iter<L>(self) -> EitherIter<L, Self> {
        EitherIter::Right(self)
    }
}

impl<'a, T: Iterator + Sized + Send + 'a> IterExt<'a> for T {}
