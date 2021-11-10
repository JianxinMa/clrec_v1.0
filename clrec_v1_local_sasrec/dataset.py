from __future__ import division
from __future__ import print_function

import copy
import random
from collections import defaultdict
from multiprocessing import Process, Queue

import numpy as np
import tensorflow as tf
from tqdm import tqdm

FLAGS = tf.app.flags.FLAGS


def load_data(fname):
    usernum = 0
    itemnum = 0
    user_all = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        user_all[u].append(i)

    for user in user_all:
        nfeedback = len(user_all[user])
        if nfeedback < 3:
            user_train[user] = user_all[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = user_all[user][:-2]
            user_valid[user] = []
            user_valid[user].append(user_all[user][-2])
            user_test[user] = []
            user_test[user].append(user_all[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, dataset, args, sess, max_num_cases=10000, testing=True,
             uniform=False):
    [train, valid, test, num_user, num_item] = copy.deepcopy(dataset)

    item_probs = np.zeros(num_item + 1)
    for _, uclicks in train.items():
        for item in uclicks:
            assert 1 <= item <= num_item
            item_probs[item] += 1.0
    item_probs /= item_probs.sum()

    num_cases, mrr = 0.0, 0.0
    hitr_1, hitr_5, hitr_10, ndcg_5, ndcg_10 = 0.0, 0.0, 0.0, 0.0, 0.0
    if num_user > max_num_cases:
        users = random.sample(range(1, num_user + 1), max_num_cases)
    else:
        users = range(1, num_user + 1)
    users = list(users)
    for i in tqdm(range(len(users)), total=len(users), ncols=70,
                  leave=False, unit='u'):
        u = users[i]
        if len(train[u]) < 1 or len(valid[u]) < 1:
            continue
        if testing and len(test[u]) < 1:
            continue
        assert len(valid[u]) == 1 and len(test[u]) == 1

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        if testing:
            seq[idx] = valid[u][0]
            idx -= 1
        for item in reversed(train[u]):
            seq[idx] = item
            idx -= 1
            if idx == -1:
                break

        rated = set(train[u] + test[u] + valid[u])
        rated.add(0)
        item_idx = [test[u][0] if testing else valid[u][0]]
        while len(item_idx) < 101:
            if uniform:
                t = np.random.randint(1, num_item + 1)
                if t not in rated:
                    item_idx.append(t)
            else:
                sampled_ids = np.random.choice(
                    num_item + 1, 101, replace=False, p=item_probs)
                sampled_ids = [x for x in sampled_ids if x not in rated]
                item_idx.extend(sampled_ids)
                rated = rated.union(set(sampled_ids))
        item_idx = item_idx[:101]

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]
        assert not np.isnan(predictions[0])
        assert not np.isinf(predictions[0])
        rank = predictions.argsort().argsort()[0]

        num_cases += 1.0
        if rank < 1:
            hitr_1 += 1.0
        if rank < 5:
            hitr_5 += 1.0
            ndcg_5 += 1.0 / np.log2(rank + 2.0)
        if rank < 10:
            hitr_10 += 1.0
            ndcg_10 += 1.0 / np.log2(rank + 2.0)
        mrr += 1.0 / (rank + 1)

    return (hitr_1 / num_cases, hitr_5 / num_cases, hitr_10 / num_cases,
            ndcg_5 / num_cases, ndcg_10 / num_cases, mrr / num_cases)


class FifoQueue:
    def __init__(self):
        self.q = []

    def extend(self, x):
        self.q.extend(x)

    def pop(self, n):
        assert len(self.q) >= n
        result = self.q[:n]
        assert len(result) == n
        self.q = self.q[n:]
        return result

    def __len__(self):
        return len(self.q)


def _sample_worker(user_train, usernum, itemnum, batch_size, maxlen,
                   uniform, result_queue, seed, queue_mode=True):
    click_streams = []
    item_probs = np.zeros(itemnum + 1)
    for _, uclicks in user_train.items():
        for item in uclicks:
            assert 1 <= item <= itemnum
            item_probs[item] += 1.0
            click_streams.append(item)
    item_probs /= item_probs.sum()

    def sample_one_seq(max_size, rated):
        i = np.random.randint(0, len(click_streams))
        neg = []
        while len(neg) < max_size and i < len(click_streams):
            if click_streams[i] not in rated:
                neg.append(click_streams[i])
            i += 1
        return neg

    neg_queue = FifoQueue()

    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            nxt = i
            idx -= 1
            if idx == -1:
                break

        rated = set(user_train[user])
        rated.add(0)
        neg = []
        while len(neg) < maxlen:
            assert not uniform
            assert queue_mode
            if uniform:
                sampled_ids = np.random.choice(
                    itemnum + 1, maxlen, replace=False)
            elif queue_mode:
                while len(neg_queue) < maxlen:
                    neg_queue.extend(sample_one_seq(maxlen, []))
                sampled_ids = neg_queue.pop(maxlen)
            else:
                sampled_ids = sample_one_seq(maxlen, rated)
                # sampled_ids = np.random.choice(
                #     itemnum + 1, maxlen, replace=False, p=item_probs)
            sampled_ids = [x for x in sampled_ids if x not in rated]
            neg.extend(sampled_ids)
            rated = rated.union(set(sampled_ids))
        neg = np.asarray(neg[:maxlen], dtype=np.int32)

        if queue_mode:
            neg_queue.extend(list(user_train[user]))

        return user, seq, pos, neg

    np.random.seed(seed)
    while True:
        one_batch = []
        for _ in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class BatchSampler(object):
    def __init__(self, user_train, usernum, itemnum, batch_size, maxlen,
                 uniform=False, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=_sample_worker, args=(
                    user_train, usernum, itemnum, batch_size, maxlen, uniform,
                    self.result_queue, np.random.randint(int(2e9))
                )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
