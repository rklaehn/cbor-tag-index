/// Like the one from itertools, but more convenient
#[cfg(test)]
pub(crate) enum EitherIter<L, R> {
    Left(L),
    Right(R),
}

#[cfg(test)]
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

    #[cfg(test)]
    fn left_iter<R>(self) -> EitherIter<Self, R> {
        EitherIter::Left(self)
    }

    #[cfg(test)]
    fn right_iter<L>(self) -> EitherIter<L, Self> {
        EitherIter::Right(self)
    }
}

impl<'a, T: Iterator + Sized + Send + 'a> IterExt<'a> for T {}

/// Create an iterator from a faillble fn
///
/// the fn will be called until it returns either Ok(None) or Err(...)
pub(crate) fn from_fallible_fn<'a, T: 'a>(
    mut f: impl FnMut() -> anyhow::Result<Option<T>> + 'a,
) -> impl Iterator<Item = anyhow::Result<T>> + 'a {
    std::iter::repeat(())
        .flat_map(move |_| match f() {
            Err(cause) => vec![Some(Err(cause)), None],
            Ok(Some(value)) => vec![Some(Ok(value))],
            Ok(None) => vec![None],
        })
        .take_while(|x| x.is_some())
        .filter_map(|x| x)
}
