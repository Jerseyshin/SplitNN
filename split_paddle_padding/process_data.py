import paddle
import numpy as np


def prepare_movielens_data(sample_size, batch_size, watch_vec_size, search_vec_size,
                           other_feat_size, dataset_size, label_actual_filepath):
    """
    prepare movielens data
    """
    watch_vecs = []
    search_vecs = []
    other_feats = []
    labels = []

    # prepare movielens data
    movie_info = paddle.dataset.movielens.movie_info()
    user_info = paddle.dataset.movielens.user_info()

    max_user_id = paddle.dataset.movielens.max_user_id()
    user_watch = np.zeros((max_user_id, watch_vec_size))
    user_search = np.zeros((max_user_id, search_vec_size))
    user_feat = np.zeros((max_user_id, other_feat_size))
    user_labels = np.zeros((max_user_id, 1))

    MOVIE_EMBED_TAB_HEIGHT = paddle.dataset.movielens.max_movie_id()
    MOVIE_EMBED_TAB_WIDTH = watch_vec_size

    JOB_EMBED_TAB_HEIGHT = paddle.dataset.movielens.max_job_id() + 1
    JOB_EMBED_TAB_WIDTH = paddle.dataset.movielens.max_job_id() + 1

    AGE_EMBED_TAB_HEIGHT = len(paddle.dataset.movielens.age_table)
    AGE_EMBED_TAB_WIDTH = len(paddle.dataset.movielens.age_table)

    GENDER_EMBED_TAB_HEIGHT = 2
    GENDER_EMBED_TAB_WIDTH = 4

    np.random.seed(1)

    MOVIE_EMBED_TAB = np.zeros((MOVIE_EMBED_TAB_HEIGHT, MOVIE_EMBED_TAB_WIDTH))
    AGE_EMBED_TAB = np.zeros((AGE_EMBED_TAB_HEIGHT, AGE_EMBED_TAB_WIDTH))
    GENDER_EMBED_TAB = np.zeros((GENDER_EMBED_TAB_HEIGHT, GENDER_EMBED_TAB_WIDTH))
    JOB_EMBED_TAB = np.zeros((JOB_EMBED_TAB_HEIGHT, JOB_EMBED_TAB_WIDTH))

    for i in range(MOVIE_EMBED_TAB_HEIGHT):
        MOVIE_EMBED_TAB[i][hash(i) % MOVIE_EMBED_TAB_WIDTH] = 1
        MOVIE_EMBED_TAB[i][hash(hash(i)) % MOVIE_EMBED_TAB_WIDTH] = 1

    for i in range(AGE_EMBED_TAB_HEIGHT):
        AGE_EMBED_TAB[i][i] = 1

    for i in range(GENDER_EMBED_TAB_HEIGHT):
        GENDER_EMBED_TAB[i][i] = 1

    for i in range(JOB_EMBED_TAB_HEIGHT):
        JOB_EMBED_TAB[i][i] = 1

    train_set_creator = paddle.dataset.movielens.train()

    pre_uid = 0
    movie_count = 0
    user_watched_movies = [[] for i in range(dataset_size)]
    for instance in train_set_creator():
        uid = int(instance[0]) - 1
        gender_id = int(instance[1])
        age_id = int(instance[2])
        job_id = int(instance[3])
        mov_id = int(instance[4]) - 1
        user_watched_movies[uid].append(mov_id)
        user_watch[uid, :] += MOVIE_EMBED_TAB[mov_id, :]
        user_labels[uid, :] = mov_id

        user_feat[uid, :] = np.concatenate((JOB_EMBED_TAB[job_id, :],
                                            GENDER_EMBED_TAB[gender_id, :],
                                            AGE_EMBED_TAB[age_id, :]))

        if uid == pre_uid:
            movie_count += 1
        else:
            user_watch[pre_uid, :] = user_watch[pre_uid, :] / movie_count
            movie_count = 1
            pre_uid = uid
    user_watch[pre_uid, :] = user_watch[pre_uid, :] / movie_count

    user_search = user_watch

    # if (os.path.exists(label_actual_filepath)):
    #     os.system('rm -rf'  + label_actual_filepath)
    # user_watched_movies_vec = pd.DataFrame(user_watched_movies)
    # user_watched_movies_vec.to_csv(label_actual_filepath, mode='a', index=False, header=0)
    # watch_filepath="./data/user_watch.csv"
    # user_watched_movies_vec = pd.DataFrame(user_watch)
    # user_watched_movies_vec.to_csv(label_actual_filepath, mode='a', index=False, header=0)

    return user_watch, user_search, user_feat, user_labels


def prepare_movielens_test_data(sample_size, batch_size, watch_vec_size, search_vec_size,
                                other_feat_size, dataset_size, label_actual_filepath):
    """
    prepare movielens data
    """
    watch_vecs = []
    search_vecs = []
    other_feats = []
    labels = []

    # prepare movielens data
    movie_info = paddle.dataset.movielens.movie_info()
    user_info = paddle.dataset.movielens.user_info()

    max_user_id = paddle.dataset.movielens.max_user_id()
    user_watch = np.zeros((max_user_id, watch_vec_size))
    user_search = np.zeros((max_user_id, search_vec_size))
    user_feat = np.zeros((max_user_id, other_feat_size))
    user_labels = np.zeros((max_user_id, 1))

    MOVIE_EMBED_TAB_HEIGHT = paddle.dataset.movielens.max_movie_id()
    MOVIE_EMBED_TAB_WIDTH = watch_vec_size

    JOB_EMBED_TAB_HEIGHT = paddle.dataset.movielens.max_job_id() + 1
    JOB_EMBED_TAB_WIDTH = paddle.dataset.movielens.max_job_id() + 1

    AGE_EMBED_TAB_HEIGHT = len(paddle.dataset.movielens.age_table)
    AGE_EMBED_TAB_WIDTH = len(paddle.dataset.movielens.age_table)

    GENDER_EMBED_TAB_HEIGHT = 2
    GENDER_EMBED_TAB_WIDTH = 4

    np.random.seed(1)

    MOVIE_EMBED_TAB = np.zeros((MOVIE_EMBED_TAB_HEIGHT, MOVIE_EMBED_TAB_WIDTH))
    AGE_EMBED_TAB = np.zeros((AGE_EMBED_TAB_HEIGHT, AGE_EMBED_TAB_WIDTH))
    GENDER_EMBED_TAB = np.zeros((GENDER_EMBED_TAB_HEIGHT, GENDER_EMBED_TAB_WIDTH))
    JOB_EMBED_TAB = np.zeros((JOB_EMBED_TAB_HEIGHT, JOB_EMBED_TAB_WIDTH))

    for i in range(MOVIE_EMBED_TAB_HEIGHT):
        MOVIE_EMBED_TAB[i][hash(i) % MOVIE_EMBED_TAB_WIDTH] = 1
        MOVIE_EMBED_TAB[i][hash(hash(i)) % MOVIE_EMBED_TAB_WIDTH] = 1

    for i in range(AGE_EMBED_TAB_HEIGHT):
        AGE_EMBED_TAB[i][i] = 1

    for i in range(GENDER_EMBED_TAB_HEIGHT):
        GENDER_EMBED_TAB[i][i] = 1

    for i in range(JOB_EMBED_TAB_HEIGHT):
        JOB_EMBED_TAB[i][i] = 1

    train_set_creator = paddle.dataset.movielens.test()

    pre_uid = 0
    movie_count = 0
    user_watched_movies = [[] for i in range(dataset_size)]
    for instance in train_set_creator():
        uid = int(instance[0]) - 1
        gender_id = int(instance[1])
        age_id = int(instance[2])
        job_id = int(instance[3])
        mov_id = int(instance[4]) - 1
        user_watched_movies[uid].append(mov_id)
        user_watch[uid, :] += MOVIE_EMBED_TAB[mov_id, :]
        user_labels[uid, :] = mov_id

        user_feat[uid, :] = np.concatenate((JOB_EMBED_TAB[job_id, :],
                                            GENDER_EMBED_TAB[gender_id, :],
                                            AGE_EMBED_TAB[age_id, :]))

        if uid == pre_uid:
            movie_count += 1
        else:
            user_watch[pre_uid, :] = user_watch[pre_uid, :] / movie_count
            movie_count = 1
            pre_uid = uid
    user_watch[pre_uid, :] = user_watch[pre_uid, :] / movie_count

    user_search = user_watch

    # if (os.path.exists(label_actual_filepath)):
    #     os.system('rm -rf'  + label_actual_filepath)
    # user_watched_movies_vec = pd.DataFrame(user_watched_movies)
    # user_watched_movies_vec.to_csv(label_actual_filepath, mode='a', index=False, header=0)
    # watch_filepath="./data/user_watch.csv"
    # user_watched_movies_vec = pd.DataFrame(user_watch)
    # user_watched_movies_vec.to_csv(label_actual_filepath, mode='a', index=False, header=0)

    return user_watch, user_search, user_feat, user_labels

if __name__=="__main__":
    watch_vec_size = 64
    search_vec_size = 64
    other_feat_size = 32
    label_path = "/Users/lizhenyu/PycharmProjects/YoutubeDNN/label.csv"
    user_watch, user_search, user_feat, user_labels = prepare_movielens_data(0, 32, watch_vec_size, search_vec_size,
                                                                                  other_feat_size, 6040, label_path)
    np.save('user_watch', user_watch)
    np.save('user_search', user_search)
    np.save('user_feat', user_feat)
    np.save('user_labels', user_labels)


