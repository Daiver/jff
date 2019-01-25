mod tests {
    #[test]
    fn test_find_best_split_mse() {
            //::gbt::foo();
            println!("Not implemented");
            let data = vec![
                vec![1.0],
                vec![3.0],
                vec![5.0],
                vec![4.0],
                vec![2.0]
            ];
            let targets = vec![10.0, 30.0, 50.0, 40.0, 20.0];
            let res = ::gbt::find_best_split_mse(data, targets, 0);
            assert!(res.feat_ind == 0);
            assert!((res.threshold - 2.5).abs() < 0.000001);
            assert!((res.score - 91.66666).abs() < 0.0001)
    }

}
