

def preprocess_data():
    import numpy as np
    import pickle

    # EMBEDDING SIZE
    with open(r"../data/citeulike/citeulike-a/vocabulary.dat") as vocabulary_file:
        embedding_size = len(vocabulary_file.readlines())

    # Create Item Matrix 
    with open(r"../data/citeulike/citeulike-a/mult.dat") as item_info_file:
        # initialize item matrix (16980 , 8000)
        item_size = len(item_info_file.readlines())
        item_bow = np.zeros((item_size , embedding_size))

        sentences = item_info_file.readlines()
        for index,sentence in enumerate(sentences):
            words = sentence.strip().split(" ")[1:]
            for word in words:
                vocabulary_index , number = word.split(":")
                item_bow[index][int(vocabulary_index)] = number

    #find user_size = 5551
    with open(r"../data/citeulike/citeulike-a/users.dat") as rating_file:
        user_size = len(rating_file.readlines())

    #initialize rating_matrix (5551 , 16980)
    import numpy as np
    rating_matrix = np.zeros((user_size , item_size))

    #build rating_matrix
    with open(r"../data/citeulike/citeulike-a/users.dat") as rating_file:
        lines = rating_file.readlines()
        for index,line in enumerate(lines):
            items = line.strip().split(" ")
            for item in items:  
                rating_matrix[index][int(item)] = 1

    with open(r'../data/citeulike/citeulike-a/item_bow.pickle', 'wb') as handle:
        pickle.dump(item_bow, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(r'../data/citeulike/citeulike-a/rating_matrix.pickle', 'wb') as handle:
        pickle.dump(rating_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)


# preprocess_data()