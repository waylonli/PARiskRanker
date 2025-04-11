import numpy as np


class OutliersFinder():
    def __init__(self, n_insts, outliers_type, start, end, cutoff, min_neg_rel=0, last_sour=False, curr_sour=False):
        self.n_insts = n_insts
        self.outliers_type = outliers_type
        self.start = start
        self.end = end
        self.cutoff = cutoff
        self.min_neg_rel = min_neg_rel # minimal doc label to consider a document as not-relevant
        self.last_sour = last_sour
        self.curr_sour = curr_sour


        self.outleirs_ids = np.zeros((n_insts, ))
        self.last_outleirs_ids = None
        self.curr_outleirs_ids = [np.zeros((n_insts, ))]

        self.iteration = 0
        self.curr_end = 0
        self.perc_counter = 0.

        if self.outliers_type not in ["neg", "pos", "all"]:
            raise "<outliers_type> must by 'neg', 'pos', or 'all'"
        
        self.mode = self._sour
        if self.last_sour:
            self.mode = self._last_sour
        if self.curr_sour:
            self.mode = self._curr_sour

    def _sour(self, y_score, data):
        if data.params['name'] == "train":
            if self.iteration >= self.start and self.iteration < self.end:
                iter_outleirs_ids = self._compute_outliers_ids(data.label, y_score, data.group)
                self.outleirs_ids[iter_outleirs_ids] += 1
                self.perc_counter += 1

        val = np.round(self.perc_counter / (self.end - self.start) * 100, decimals=2)
        return f". mode: SOUR. block [{self.start}, {self.end}). done %", val

    def _curr_sour(self, y_score, data):
        if data.params['name'] == "train":
            if self.curr_end < len(self.end) - 1 and self.iteration >= self.end[self.curr_end]:
                self.curr_end += 1
                self.curr_outleirs_ids.append(np.copy(self.curr_outleirs_ids[self.curr_end - 1]))
                
            if self.iteration >= self.start and self.iteration < self.end[self.curr_end]:
                iter_outleirs_ids = self._compute_outliers_ids(data.label, y_score, data.group)
                self.curr_outleirs_ids[self.curr_end][iter_outleirs_ids] += 1
                self.perc_counter += 1

        val = np.round(self.perc_counter / (self.end[-1] - self.start) * 100, decimals=2)
        return f". mode: curr-SOUR. block [{self.start}, {self.end[self.curr_end]}). done %", val

    def _last_sour(self, y_score, data):
        if data.params['name'] == "train":
            if self.iteration == self.end - 1:
                self.last_outleirs_ids = self._compute_outliers_ids(data.label, y_score, data.group)

        return ". mode: last-SOUR. last iter", self.end

    def __call__(self, y_score, data):
        string, val = self.mode(y_score, data)
        self.iteration += 1
        return f"outliers{string}", val, True

    # computes positive and negative outlier documents' IDs
    def _compute_outliers_ids(self, y_true, y_score, qs_len):
        n_docs = y_score.shape[0]
        idx_docs = np.arange(n_docs)
        cum = np.cumsum(qs_len)[:-1]

        ids_list = []
        for labels, query, idx in zip(np.array_split(y_true,cum), np.array_split(y_score,cum), np.array_split(idx_docs,cum)):
            size = len(query)
            cut = min(self.cutoff, size)
            rk = len(query) - 1 - np.argsort(query[::-1], kind='stable')[::-1] # stable argsort in descending order

            labels_sorted = labels[rk]
            idx_sorted = idx[rk]
            pos_idx = np.arange(size)
            
            # gets negative outlier documents' IDs
            if self.outliers_type != "pos":
                idx_stop = min(cut, np.where(labels_sorted > self.min_neg_rel)[0][-1]) if np.sum(labels_sorted > self.min_neg_rel) else -1
                neg_outliers_ids = idx_sorted[np.where((labels_sorted <= self.min_neg_rel) * (pos_idx < idx_stop))[0]]
                ids_list.append(neg_outliers_ids)

            # gets positive outlier documents' IDs
            if self.outliers_type != "neg":
                idx_max = max(cut, np.argmin(labels_sorted)) if np.sum(labels_sorted <= self.min_neg_rel) else len(labels_sorted)
                pos_outliers_ids = idx_sorted[np.where((labels_sorted > self.min_neg_rel) * (pos_idx >= idx_max))[0]]
                ids_list.append(pos_outliers_ids)

        return np.concatenate(ids_list)
    
    # gets consistent / frequent / last iteration outlier documents' IDs in the interval [start, end). NOTE: counting from zero.   
    def get_outliers_ids(self, p_sour=1, last_sour=False, curr_sour=False):
        if last_sour:
            return [self.last_outleirs_ids]

        if not curr_sour:
            return [np.where(self.outleirs_ids >= int(p_sour * (self.end - self.start)))[0]]

        empty_docs_idx = np.array([], dtype=int)
        idx_to_remove = np.zeros(self.curr_outleirs_ids[0].shape)
        list_outleirs_ids = [np.where(self.curr_outleirs_ids[i] >= int(p_sour * (end - self.start)))[0] for i, end in enumerate(self.end)]
        list_outleirs_ids.append(empty_docs_idx)

        final_list = []
        for i, idx in enumerate(list_outleirs_ids):
            idx_to_remove[idx] += 1
            final_list.append(np.where(idx_to_remove > i)[0])

        return final_list
    
    def get_outliers_ids_from_score(self, y_true, y_score, qs_len):
        return self._compute_outliers_ids(y_true, y_score, qs_len)