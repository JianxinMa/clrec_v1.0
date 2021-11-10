import numpy as np

# [samplestore] is a sub-project of [graph-learn](https://github.com/alibaba/graph-learn),
# which is for preprocessing graph data, e.g., the user-item bipartite click graph of a
# recommender system. However, the samplestore API is now deprecated, and its functionality
# has been integrated into graph-learn. Please consider remove the samplestore related code
# and use your own way to prepare the data at your convenience.
import samplestore as ss
import tensorflow as tf


class GraphInput(object):
    def __init__(self, FLAGS=None):
        print('u-i graph input')
        self.FLAGS = FLAGS

        time_bucket = self.FLAGS.time_buckets
        self.time_bucket = np.array(time_bucket.split(","), dtype=np.int32)

        self.features = self.init_input(FLAGS)

    def init_server(self):
        batch_size = self.FLAGS.batch_size

        edge_path = self.FLAGS.tables.split(',')[0]

        # read the user-item click graph (ui_src = User-Item data SouRCe)
        self.ui_src = ss.GraphSource(path=edge_path, alias="u-i")

        worker_count = len(self.FLAGS.worker_hosts.split(","))

        # allow a non-chief worker of the distributed cluster to restart if it fails
        ss.set_enable_failover(True)

        # padding with zero when the number of neighbors is less than the specified number
        ss.set_default_neighbor_id(0)

        # ss.set_batch_num_for_shuffle(num)
        # ss.set_datainit_threadnum(32)
        # ss.set_datainit_batchsize(2048*100)

        # start the samplestore service
        self.server = ss.Server(server_id=self.FLAGS.task_index, server_count=worker_count)
        print('server.start')
        self.server.start()

        # init data from source
        print('server.init')
        self.server.init(graph_source=[self.ui_src])

        print('finish server.init')

        self.ui_graph = ss.Graph(self.ui_src)

        # iterator for sampling a random set of users (u_node_iter = user node iterator)
        self.u_node_iter = ss.IterateNodeSampler(batch_size=batch_size)

        # iterator for sampling a random set of user-item edges (user-item clicks)
        self.edge_sampler = ss.IterateEdgeSampler(batch_size=batch_size, shuffle=True)

        if self.FLAGS.mode == 'train':
            # This samplers will return a list of neighbors of a given user, i.e., the items clicked by the user.
            # The list will be of size self.FLASG.hist_max, self.FLASG.hist_max = 20 by default.
            # The sampled clicks in the list are the consecutive clicks made by the user, i.e., they are in a
            # sliding window of size self.FLASG.hist_max.
            self.hop_u_sampler = ss.WindowNeighborSampler(high_size=0, low_size=self.FLAGS.hist_max)
        else:
            # This samplers will return a list of neighbors of a given user, i.e., the items clicked by the user.
            # The clicks in the list are the latest self.FLASG.hist_max (20 by default) clicks made by the user.
            self.hop_u_sampler = ss.WindowNeighborSampler(high_size=self.FLAGS.hist_max + 1, low_size=0)

    def init_input(self, FLAGS):
        self.u_ids = tf.placeholder(shape=[None], dtype=tf.string)  # [B], batch of user_ids
        self.i_ids = tf.placeholder(shape=[None], dtype=tf.string)  # [B], batch of positive item_ids
        self.item = tf.placeholder(shape=[None, FLAGS.hist_max], dtype=tf.string)  # [B,T], items clicked by the users
        self.nbr_mask = tf.placeholder(shape=[None], dtype=tf.int64)  # [B], the number of clicks made by each user
        self.samples_mask = tf.placeholder(shape=[None], dtype=tf.int64)  # [B], =0 means to ignore this sample

        # [B,T], the time when the click happens
        self.tb_feats = tf.placeholder(shape=[None, FLAGS.hist_max], dtype=tf.int64)

        print('finish placeholder init.')
        return {
            'uid': self.u_ids,
            'iid': self.i_ids,
            'item': self.item,
            'nbr_mask': self.nbr_mask,
            'samples_mask': self.samples_mask,
            'tb_feats': self.tb_feats
        }

    def get_tb_feats(self, target_weights, neights_weights):
        tb_feats = []
        # compuate time bucket
        for idx in range(np.shape(target_weights)[0]):
            cur_t = target_weights[idx]

            low_idx = idx * self.FLAGS.hist_max
            high_idx = (idx + 1) * self.FLAGS.hist_max

            hist_t = [(cur_t - i) // 60 for i in neights_weights[low_idx:high_idx]]

            hist_t = [np.sum(i >= self.time_bucket) for i in hist_t]

            tb_feats.append(hist_t)

        tb_feats = np.array(tb_feats, dtype=np.int64)

        return tb_feats

    def _next_sample(self, graph):
        # sample edges, each edge is of the form <user_id (src_id), clicked_item_id (dst_id)>
        edges = self.edge_sampler.get(graph, with_attr=False)

        # sample the latest self.FLAGS.hist_max clicks made by the user (src_id) before the specified click (dst_id)
        src_nbrs = self.hop_u_sampler.get([graph], edges.src_ids, edges.dst_ids, with_attr=False)

        # the corresponding timestamp of the click (<src_id, dst_id>)
        target_weights = np.array(graph.get_weights(edges.src_ids, edges.dst_ids, default_weight=0)).flatten()

        # the corresponding timestamp of the latest self.FLAGS.hist_max clicks
        neights_weights = np.array(
            graph.get_weights(np.tile(np.reshape(edges.src_ids, [-1, 1]), [1, self.FLAGS.hist_max]).flatten(),
                              src_nbrs.hop(1).ids.flatten(), default_weight=0)).flatten()

        # print("target_weights", target_weights)
        # print("neights_weights", neights_weights)
        #
        # print("time bucket", self.time_bucket)

        # time bucket features
        tb_feats = self.get_tb_feats(target_weights, neights_weights)

        return src_nbrs, edges, tb_feats

    def _next_user(self, graph):
        nodes = self.u_node_iter.get(graph, with_attr=False)
        src_ids = nodes.ids

        # sample the latest self.FLAGS.hist_max clicks made by the user (src_id)
        src_nbrs = self.hop_u_sampler.get([graph], src_ids, None, with_attr=False)
        nbrs_ids = np.reshape(src_nbrs.hop(1).ids, [-1, self.FLAGS.hist_max + 1])

        lst_ids = np.reshape(nbrs_ids[:, -1], -1)
        front_ids = nbrs_ids[:, :-1]

        nbr_mask = np.reshape(src_nbrs.hop(1).real_nbr_count, [-1]) - 1

        # the corresponding timestamp of the click (<src_id, dst_id>)
        target_weights = np.array(graph.get_weights(src_ids, lst_ids, default_weight=0)).flatten()

        # the corresponding timestamp of the latest self.FLAGS.hist_max clicks
        neights_weights = np.array(
            graph.get_weights(np.tile(np.reshape(src_ids, [-1, 1]), [1, self.FLAGS.hist_max]).flatten(),
                              front_ids.flatten(), default_weight=0)).flatten()

        # time bucket features
        tb_feats = self.get_tb_feats(target_weights, neights_weights)

        return src_ids, front_ids, nbr_mask, tb_feats

    def _next_item(self, graph):
        nodes = self.u_node_iter.get(graph, with_attr=False)
        src_ids = nodes.ids

        return src_ids

    def _feed_next_sample(self, pos_sample_table):
        src_nbrs, edges, tb_feats = self._next_sample(pos_sample_table)

        res = dict()

        src_ids = np.reshape(edges.src_ids, [-1])
        dst_ids = np.reshape(edges.dst_ids, [-1])

        nbr_mask = np.reshape(src_nbrs.hop(1).real_nbr_count, [-1])
        nbr_ids = np.reshape(src_nbrs.hop(1).ids, [-1, self.FLAGS.hist_max])
        samples_mask = np.where(nbr_mask >= 3, 1, 0)
        nbr_mask = nbr_mask * samples_mask
        tb_feats = np.reshape(tb_feats, [-1, self.FLAGS.hist_max])

        # print("---------------input------------------")
        # print("u_ids", src_ids)
        # print("i_ids", dst_ids)
        # print("item", nbr_ids)
        # print("nbr_mask", nbr_mask)
        # print("samples_mask", samples_mask)
        # print("tb_feats", tb_feats)
        # print("--------------------------------------")

        res[self.u_ids] = src_ids.astype(np.str)
        res[self.i_ids] = dst_ids.astype(np.str)
        res[self.item] = nbr_ids.astype(np.str)
        res[self.nbr_mask] = nbr_mask
        res[self.samples_mask] = samples_mask
        res[self.tb_feats] = tb_feats

        return res

    def stop(self):
        self.server.stop()

    def feed_next_item(self):
        src_ids = self._next_item(self.ui_graph)

        res = dict()
        res[self.i_ids] = src_ids.astype(np.str)

        return res

    def feed_next_user(self):
        src_ids, nbr_ids, nbr_mask, tb_feats = self._next_user(self.ui_graph)

        # print(tb_feats)
        res = dict()
        res[self.u_ids] = src_ids.astype(np.str)
        res[self.nbr_mask] = nbr_mask
        res[self.item] = nbr_ids.astype(np.str)
        res[self.tb_feats] = tb_feats

        return res

    def feed_next_sample_train(self):
        return self._feed_next_sample(self.ui_graph)

    def feed_next_sample_eval(self):
        return self._feed_next_sample(self.ui_graph_eval)
