#[burn_tensor_testgen::testgen(cumsum_dim)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Int, Tensor};

    #[test]
    fn should_cumsum_over_dim() {
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &Default::default());

        let dim = 1;

        let data_actual = tensor.cumsum_dim(dim).into_data();
        dbg!(data_actual.clone());
        let data_expected = Data::from([[0.0, 1.0, 3.0], [3.0, 7.0, 12.0], [6.0, 13.0, 21.0]]);

        data_expected.assert_approx_eq(&data_actual, 3);
    }

    // #[test]
    // #[should_panic(expected = "attempt to add with overflow")]
    // fn should_panic_on_int_overflow() {
    //     let data = Data::from([[std::i64::MAX, 1], [std::i64::MAX, 2]]);
    //     let tensor = Tensor::<TestBackend, 2, Int>::from_data(data, &Default::default());

    //     let dim = 1;

    //     // This should trigger an overflow panic when cumulating sums
    //     let _ = tensor.cumsum_dim(dim).into_data();
    // }
}
