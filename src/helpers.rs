pub trait FetchIncrement: Sized {
    fn fetch_inc(&mut self) -> Self;
}

impl FetchIncrement for u32 {
    fn fetch_inc(&mut self) -> Self {
        let val = *self;
        *self += 1;
        val
    }
}
