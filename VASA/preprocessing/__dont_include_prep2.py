def global_moran(grouped, cols: List[str], date_col: str, group_col: str, map_data: pd.DataFrame, map_group_col: str, limit=None, sig=0.05):
    if not isinstance(cols, list):
        cols = [cols]

    output = pd.DataFrame(
        columns=['date', *[i + "_est" for i in cols], *[i + "_p" for i in cols]]
    )
    W = lps.weights.Queen(map_data["geometry"])
    W.transform = 'r'

    for i, (date, group) in tqdm(enumerate(grouped), total=min(limit, len(grouped)) if limit else len(grouped)):

        if limit and i == limit:
            break 

        ordered = pd.merge(
            map_data, group, left_on=map_group_col, right_on=group_col, how='left'
        )

        row = {
            'date': [min(group['date_start']), max(group['date_end'])] if ('date_start' in ordered.columns) else min(group[date_col])
        }

        for j, col in enumerate(cols):
            values = ordered[[map_group_col, col, "geometry"]]
            mi = Moran(values[col], W, transformation='r', two_tailed=False, permutations=n_permutations(values))
            row[col + "_est"] = mi.I
            row[col + "_p"] = mi.p_sim
        output = output.append(row, ignore_index=True)

    return (output, map_data[map_group_col])
