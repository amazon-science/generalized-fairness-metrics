from collections.abc import MutableMapping
import pandas as pd
import regex as re


class LexiconsDict(MutableMapping):
    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        # e.g. gender_adj_basic (we don't store those directly, but under
        # 'gender' as a data frame -- for memory efficiency and this also
        # allows to directly access that dataframe through lexicons['gender'])
        # e.g. to check what properties it supports, or query it directly
        if key not in self.store and "_" in key and \
                key.split("_")[0] in self.store and \
                type(self.store[key.split("_")[0]]) == pd.DataFrame:
            parts = key.split("_")
            term = parts[0]
            df = self.store[term]

            # TODO: this assumes the sets of possible values are different
            # for each of the properties, which is not always the case
            val2property = {}
            for p in df.columns:
                for val in df[p].unique():
                    if type(val) == int:
                        continue
                    val2property[val] = p

            try:
                # if this remains None then we return index (term)
                form_to_ret = None
                for pval in parts:
                    if pval == term:
                        continue

                    if pval == "all":
                        # don't filter anything, return all terms
                        break

                    # if there is a RANK in the df then allow to query
                    # for top x terms, e.g. names_top100
                    top_match = re.match(r'^top([0-9]+)$', pval)

                    # when another column's value is to be returned instead of
                    # the index (e.g. another form of a verb)
                    if pval.upper() in df.columns:
                        form_to_ret = pval.upper()
                    elif top_match:
                        if "RANK" in df.columns:
                            rank = int(top_match.group(1))
                            df = df[df["RANK"] <= rank]
                    else:
                        prop = val2property[pval]
                        df = df[df[prop] == pval]
            except Exception:
                raise KeyError

            # indexed by "TERM" (see read_csv_terms in fill_the_lexicon)
            if not form_to_ret:
                return list(df.index)
            else:
                return list(df[form_to_ret])
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)
