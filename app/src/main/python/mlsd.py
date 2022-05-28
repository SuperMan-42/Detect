import numpy as np


def detect(data):
    params = {'score': 0.06,
              'outside_ratio': 0.28,
              'inside_ratio': 0.45,
              'w_overlap': 0.0,
              'w_degree': 1.95,
              'w_length': 0.0,
              'w_area': 1.86,
              'w_center': 0.14}

    segments_list = []
    for i in range(data.size()):
        box = data.get(i).getBox_origin()
        segments_list.append([box.left, box.top, box.right, box.bottom])
    new_segments = np.array(segments_list)  # (x1, y1, x2, y2)
    start = new_segments[:, :2]  # (x1, y1)
    end = new_segments[:, 2:]  # (x2, y2)
    new_centers = (start + end) / 2.0
    diff = start - end
    dist_segments = np.sqrt(np.sum(diff ** 2, axis=-1))

    # ax + by = c
    a = diff[:, 1]
    b = -diff[:, 0]
    c = a * start[:, 0] + b * start[:, 1]
    pre_det = a[:, None] * b[None, :]
    det = pre_det - np.transpose(pre_det)

    pre_inter_y = a[:, None] * c[None, :]
    inter_y = (pre_inter_y - np.transpose(pre_inter_y)) / (det + 1e-10)
    pre_inter_x = c[:, None] * b[None, :]
    inter_x = (pre_inter_x - np.transpose(pre_inter_x)) / (det + 1e-10)
    inter_pts = np.concatenate([inter_x[:, :, None], inter_y[:, :, None]], axis=-1).astype('int32')

    # 3. get corner information
    # 3.1 get distance

    dist_inter_to_segment1_start = np.sqrt(np.sum(((inter_pts - start[:, None, :]) ** 2), axis=-1,
                                                  keepdims=True))  # [n_batch, n_batch, 1]
    dist_inter_to_segment1_end = np.sqrt(np.sum(((inter_pts - end[:, None, :]) ** 2), axis=-1,
                                                keepdims=True))  # [n_batch, n_batch, 1]
    dist_inter_to_segment2_start = np.sqrt(np.sum(((inter_pts - start[None, :, :]) ** 2), axis=-1,
                                                  keepdims=True))  # [n_batch, n_batch, 1]
    dist_inter_to_segment2_end = np.sqrt(np.sum(((inter_pts - end[None, :, :]) ** 2), axis=-1,
                                                keepdims=True))  # [n_batch, n_batch, 1]
    # sort ascending
    dist_inter_to_segment1 = np.sort(
        np.concatenate([dist_inter_to_segment1_start, dist_inter_to_segment1_end], axis=-1),
        axis=-1)  # [n_batch, n_batch, 2]
    dist_inter_to_segment2 = np.sort(
        np.concatenate([dist_inter_to_segment2_start, dist_inter_to_segment2_end], axis=-1),
        axis=-1)  # [n_batch, n_batch, 2]

    # 3.2 get degree
    inter_to_start = new_centers[:, None, :] - inter_pts
    deg_inter_to_start = np.arctan2(inter_to_start[:, :, 1], inter_to_start[:, :, 0]) * 180 / np.pi
    deg_inter_to_start[deg_inter_to_start < 0.0] += 360
    inter_to_end = new_centers[None, :, :] - inter_pts
    deg_inter_to_end = np.arctan2(inter_to_end[:, :, 1], inter_to_end[:, :, 0]) * 180 / np.pi
    deg_inter_to_end[deg_inter_to_end < 0.0] += 360

    # rename variables
    deg1_map, deg2_map = deg_inter_to_start, deg_inter_to_end
    # sort deg ascending
    deg_sort = np.sort(np.concatenate([deg1_map[:, :, None], deg2_map[:, :, None]], axis=-1),
                       axis=-1)

    deg_diff_map = np.abs(deg1_map - deg2_map)
    # we only consider the smallest degree of intersect
    deg_diff_map[deg_diff_map > 180] = 360 - deg_diff_map[deg_diff_map > 180]

    # define available degree range
    deg_range = [60, 120]

    corner_dict = {corner_info: [] for corner_info in range(4)}
    inter_points = []
    for i in range(inter_pts.shape[0]):
        for j in range(i + 1, inter_pts.shape[1]):
            # i, j > line index, always i < j
            x, y = inter_pts[i, j, :]
            deg1, deg2 = deg_sort[i, j, :]
            deg_diff = deg_diff_map[i, j]

            check_degree = deg_diff > deg_range[0] and deg_diff < deg_range[1]

            outside_ratio = params['outside_ratio']  # over ratio >>> drop it!
            inside_ratio = params['inside_ratio']  # over ratio >>> drop it!
            check_distance = ((dist_inter_to_segment1[i, j, 1] >= dist_segments[i] and \
                               dist_inter_to_segment1[i, j, 0] <= dist_segments[
                                   i] * outside_ratio) or \
                              (dist_inter_to_segment1[i, j, 1] <= dist_segments[i] and \
                               dist_inter_to_segment1[i, j, 0] <= dist_segments[
                                   i] * inside_ratio)) and \
                             ((dist_inter_to_segment2[i, j, 1] >= dist_segments[j] and \
                               dist_inter_to_segment2[i, j, 0] <= dist_segments[
                                   j] * outside_ratio) or \
                              (dist_inter_to_segment2[i, j, 1] <= dist_segments[j] and \
                               dist_inter_to_segment2[i, j, 0] <= dist_segments[j] * inside_ratio))

            if check_degree and check_distance:
                corner_info = None

                if (deg1 >= 0 and deg1 <= 45 and deg2 >= 45 and deg2 <= 120) or \
                        (deg2 >= 315 and deg1 >= 45 and deg1 <= 120):
                    corner_info, color_info = 0, 'blue'
                elif (deg1 >= 45 and deg1 <= 125 and deg2 >= 125 and deg2 <= 225):
                    corner_info, color_info = 1, 'green'
                elif (deg1 >= 125 and deg1 <= 225 and deg2 >= 225 and deg2 <= 315):
                    corner_info, color_info = 2, 'black'
                elif (deg1 >= 0 and deg1 <= 45 and deg2 >= 225 and deg2 <= 315) or \
                        (deg2 >= 315 and deg1 >= 225 and deg1 <= 315):
                    corner_info, color_info = 3, 'cyan'
                else:
                    corner_info, color_info = 4, 'red'  # we don't use it
                    continue

                corner_dict[corner_info].append([x, y, i, j])
                inter_points.append([x, y])

    square_list = []
    connect_list = []
    segments_list = []
    for corner0 in corner_dict[0]:
        for corner1 in corner_dict[1]:
            connect01 = False
            for corner0_line in corner0[2:]:
                if corner0_line in corner1[2:]:
                    connect01 = True
                    break
            if connect01:
                for corner2 in corner_dict[2]:
                    connect12 = False
                    for corner1_line in corner1[2:]:
                        if corner1_line in corner2[2:]:
                            connect12 = True
                            break
                    if connect12:
                        for corner3 in corner_dict[3]:
                            connect23 = False
                            for corner2_line in corner2[2:]:
                                if corner2_line in corner3[2:]:
                                    connect23 = True
                                    break
                            if connect23:
                                for corner3_line in corner3[2:]:
                                    if corner3_line in corner0[2:]:
                                        square_list.append(
                                            corner0[:2] + corner1[:2] + corner2[:2] + corner3[:2])
                                        connect_list.append(
                                            [corner0_line, corner1_line, corner2_line,
                                             corner3_line])
                                        segments_list.append(
                                            corner0[2:] + corner1[2:] + corner2[2:] + corner3[2:])

    def check_outside_inside(segments_info, connect_idx):
        # return 'outside or inside', min distance, cover_param, peri_param
        if connect_idx == segments_info[0]:
            check_dist_mat = dist_inter_to_segment1
        else:
            check_dist_mat = dist_inter_to_segment2

        i, j = segments_info
        min_dist, max_dist = check_dist_mat[i, j, :]
        connect_dist = dist_segments[connect_idx]
        if max_dist > connect_dist:
            return 'outside', min_dist, 0, 1
        else:
            return 'inside', min_dist, -1, -1

    try:
        map_size = 256
        squares = np.array(square_list).reshape([-1, 4, 2])
        score_array = []
        connect_array = np.array(connect_list)
        segments_array = np.array(segments_list).reshape([-1, 4, 2])

        # get degree of corners:
        squares_rollup = np.roll(squares, 1, axis=1)
        squares_rolldown = np.roll(squares, -1, axis=1)
        vec1 = squares_rollup - squares
        normalized_vec1 = vec1 / (np.linalg.norm(vec1, axis=-1, keepdims=True) + 1e-10)
        vec2 = squares_rolldown - squares
        normalized_vec2 = vec2 / (np.linalg.norm(vec2, axis=-1, keepdims=True) + 1e-10)
        inner_products = np.sum(normalized_vec1 * normalized_vec2, axis=-1)  # [n_squares, 4]
        squares_degree = np.arccos(inner_products) * 180 / np.pi  # [n_squares, 4]

        # get square score
        overlap_scores = []
        degree_scores = []
        length_scores = []

        for connects, segments, square, degree in zip(connect_array, segments_array, squares,
                                                      squares_degree):
            ###################################### OVERLAP SCORES
            cover = 0
            perimeter = 0
            # check 0 > 1 > 2 > 3
            square_length = []

            for start_idx in range(4):
                end_idx = (start_idx + 1) % 4

                connect_idx = connects[start_idx]  # segment idx of segment01
                start_segments = segments[start_idx]
                end_segments = segments[end_idx]

                start_point = square[start_idx]
                end_point = square[end_idx]

                # check whether outside or inside
                start_position, start_min, start_cover_param, start_peri_param = check_outside_inside(
                    start_segments, connect_idx)
                end_position, end_min, end_cover_param, end_peri_param = check_outside_inside(
                    end_segments, connect_idx)

                cover += dist_segments[
                             connect_idx] + start_cover_param * start_min + end_cover_param * end_min
                perimeter += dist_segments[
                                 connect_idx] + start_peri_param * start_min + end_peri_param * end_min

                square_length.append(dist_segments[
                                         connect_idx] + start_peri_param * start_min + end_peri_param * end_min)

            overlap_scores.append(cover / perimeter)
            ######################################
            ###################################### DEGREE SCORES
            deg0, deg1, deg2, deg3 = degree
            deg_ratio1 = deg0 / deg2
            if deg_ratio1 > 1.0:
                deg_ratio1 = 1 / deg_ratio1
            deg_ratio2 = deg1 / deg3
            if deg_ratio2 > 1.0:
                deg_ratio2 = 1 / deg_ratio2
            degree_scores.append((deg_ratio1 + deg_ratio2) / 2)
            ######################################
            ###################################### LENGTH SCORES
            len0, len1, len2, len3 = square_length
            len_ratio1 = len0 / len2 if len2 > len0 else len2 / len0
            len_ratio2 = len1 / len3 if len3 > len1 else len3 / len1
            length_scores.append((len_ratio1 + len_ratio2) / 2)
            ######################################

        overlap_scores = np.array(overlap_scores)
        overlap_scores /= np.max(overlap_scores)

        degree_scores = np.array(degree_scores)
        # degree_scores /= np.max(degree_scores)

        length_scores = np.array(length_scores)

        ###################################### AREA SCORES
        area_scores = np.reshape(squares, [-1, 4, 2])
        area_x = area_scores[:, :, 0]
        area_y = area_scores[:, :, 1]
        correction = area_x[:, -1] * area_y[:, 0] - area_y[:, -1] * area_x[:, 0]
        area_scores = np.sum(area_x[:, :-1] * area_y[:, 1:], axis=-1) - np.sum(
            area_y[:, :-1] * area_x[:, 1:], axis=-1)
        area_scores = 0.5 * np.abs(area_scores + correction)
        area_scores /= (map_size * map_size)  # np.max(area_scores)
        ######################################

        ###################################### CENTER SCORES
        centers = np.array([[256 // 2, 256 // 2]], dtype='float32')  # [1, 2]
        # squares: [n, 4, 2]
        square_centers = np.mean(squares, axis=1)  # [n, 2]
        center2center = np.sqrt(np.sum((centers - square_centers) ** 2, axis=1))
        center_scores = center2center / (map_size / np.sqrt(2.0))

        score_w = [0.0, 1.0, 10.0, 0.5, 1.0]
        score_array = params['w_overlap'] * overlap_scores \
                      + params['w_degree'] * degree_scores \
                      + params['w_area'] * area_scores \
                      - params['w_center'] * center_scores \
                      + params['w_length'] * length_scores

        best_square = []

        sorted_idx = np.argsort(score_array)[::-1]
        score_array = score_array[sorted_idx]
        squares = squares[sorted_idx]

    except Exception as e:
        pass

    return squares.tolist()
