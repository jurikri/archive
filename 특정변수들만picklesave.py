def pickle_save():
    import pickle

    # 변수들을 딕셔너리에 저장
    data_to_save = {
        'div_img1': div_img1,  # Z_without_nan 대신 예제 데이터의 길이를 사용합니다.
        'divnum': divnum,  # 예제 값
    }
    
    # 피클로 저장
    file_path = "saved_data.pkl"
    with open(file_path, 'wb') as file:
        pickle.dump(data_to_save, file)
        
