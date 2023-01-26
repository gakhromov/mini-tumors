import numpy as np
from scipy.ndimage import label
from sklearn.mixture import GaussianMixture
import math
import nd2
import cv2
from tqdm import tqdm


def create_circular_mask(h, w, center, radius):
    '''
    Helper function to mask out cirular region of 2D array
    '''
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def return_overlapping_set(vertex, adj_matrix):
    '''
    Helper function to return set of all circles overlapping with a given circle by searching the 
    overlapping graphs adjacency matrix in a BFS manner
    '''
    # Put initial adjacent vertecies in set and populate queue 
    queue = list(np.nonzero(adj_matrix[vertex]))[0]
    ol_set = [vertex]
    
    # while queue is not empty
    while len(queue) > 0:
        # pop vertex
        vert = queue[0]
        queue = queue[1:]
        
        # Ad vertex to visited set
        ol_set.append(vert)
        
        # Get adjacent to new vertes
        vert_adj = list(np.nonzero(adj_matrix[vert]))[0]
    return ol_set

def circle_rect(rectangle):
    '''
    Enscribe a circle around a rectangle
    '''
    a = abs(rectangle[0][0] - rectangle[0][1])
    b = abs(rectangle[1][0] - rectangle[1][1])
    r = 0.5 * math.sqrt(a**2 + b **2)
    c = (rectangle[0][1] - a//2, rectangle[1][1] - b//2)
    return [c, r]


def circle_weight(circle, points):
    '''
    Compute the prportion of a circle (in an integer grid) that a point cloud covers.
    Used as a metric of circlularity
    '''
    # Get how many points should be in the circle
    
    # we could easily approximate this
    denom = round(circle[1]**2 * np.pi) // 1
    # Return fraction represented in points
    return len(points) / denom

def clean_satellites(image, verbose=True, brightness_threshold_percentile=99, elimination_threshold=50, overlap_padding=20, circle_mask_padding=10):
    '''
    Apply satellite dropplet cleaning technique to image (grayscale array). 
    Techinque should be applied to the large images and will not work properly with individual droplets

    verbose                         <--- Report progress
    brightness_threshold_percentile <--- 100 - percetnage of brightest points to consider
    elimination_threshold           <--- minimum size of contiguous bright region to be considered when checking for sattelites, 
                                         smaller regions will be automatically discardeds
    overlap_padding                 <--- padd circles when checking for overlap, in theory this should be large enough to account 
                                         for dark bands cutting through sattelite dropplets
    circle_mask_padding             <--- enlarge border of circle used to mask out a sattelite dropplet. Should be thickness of dark rim of sattelite
    '''
    #TODO: Sanity checks

    # Brightness threshold percentile of 99 percent means top 1% of brightest points will be considered
    if verbose:
        print('1/6 | Determining brightness threshold and masking brightest points')
    sample = np.random.choice(a=image.flatten(), size=1000000)
    brightness_threshold = np.percentile(sample, brightness_threshold_percentile)

    # Create mask of brightest points in image
    mask = np.zeros(image.shape)
    mask[image > brightness_threshold] = 1

    # Get contiguous objects (regions of 1's) in mask
    if verbose:
        print('2/6 | Computing objects')
    labeled_features, num_features = label(mask)

    # Enscribe features with boxes
    if verbose:
        print('3/6 | Determining object bounding boxes and circles')
    feature_squares = []
    feature_points = []

    # X and Y naming is a bit wierd here
    for feat in tqdm(range(1, num_features + 1)):
        feat_coords = np.argwhere(labeled_features == feat)
        #TODO: Do something to remove features that are only a couple of pixels
        if len(feat_coords) > elimination_threshold:
            feature_points.append(feat_coords)
            x_coords = feat_coords[:,0]
            y_coords = feat_coords[:,1]
            x_axis = [min(x_coords), max(x_coords)]
            y_axis = [min(y_coords), max(y_coords)]
            feature_squares.append([y_axis, x_axis])
    
    # Enscribe circles around identified rectangular regions
    feature_circles = []

    for rect in feature_squares:
        feature_circles.append(circle_rect(rect))

    # Determine threshold of circle-similarity to seperate sattelites from dropplet edge segments
    if verbose:
        print('4/6 | Computing circle similarity threshold and circle similarity')
    scores = []
    for i in tqdm(range(len(feature_circles))):
        scores.append(circle_weight(feature_circles[i], feature_points[i]))

    gm = GaussianMixture(n_components=2, random_state=0)

    gm.fit(np.array(scores).reshape(-1, 1))

    x_den = np.linspace(0, 1, num=100)
    y_lab = gm.predict(x_den.reshape(-1, 1))

    decision_boundary = np.where(y_lab[:-1] != y_lab[1:])[0][0] + 2

    circle_weight_threshold = decision_boundary / 100

    mask_cw = []
    fil_circles = []
    fil_points = []

    for i in range(len(feature_circles)):
        score = scores[i]
        if score >= circle_weight_threshold:
            fil_circles.append(feature_circles[i])
            fil_points.append(feature_points[i])

    
    # Merge overlapping segments into larger based on padded circle collision
    # Determine threshold of circle-similarity to seperate sattelites from dropplet edge segments
    if verbose:
        print('5/6 | Merging overlapping sattelites')
    intersect_matrix = np.zeros((len(fil_circles), len(fil_circles)))

    for x in tqdm(range(len(fil_circles))):
        #only need to compute half and no diagonals
        for y in range(len(fil_circles)):
            if x == y:
                continue
            # Two ifcles intersect if distance between center points is less than sum of radii
            dR1R2 = np.linalg.norm(np.subtract(fil_circles[x][0], fil_circles[y][0]))
            intersect_matrix[x][y] = (dR1R2 <= fil_circles[x][1] + fil_circles[y][1] + overlap_padding)
            intersect_matrix[y][x] = (dR1R2 <= fil_circles[x][1] + fil_circles[y][1]+ overlap_padding)


    # Get sets of overlapping circles
    ol_circles = []
    ol_points = []
    ol_set = []
    considered_verticies = []

    # Copy intersect matrx
    ol_intersect = intersect_matrix

    for circ_index in tqdm(range(len(fil_circles))):
        # If circle not overlapping with anyone
        if sum(ol_intersect[circ_index]) == 0:
            # Just put it in the output array
            ol_circles.append(fil_circles[circ_index])
            ol_points.append(fil_points[circ_index])
        else:
            # Theres some overlapping afoot!
        
            # If vertex already considered, ignore
            if circ_index in considered_verticies:
                continue
            # Get indecies of all overlapping circles
            new_ol_set = return_overlapping_set(circ_index, ol_intersect)
        
            ol_set.append(new_ol_set)
            considered_verticies = considered_verticies + new_ol_set
        
            # Merge point clouds in new_ol_set
            new_points = fil_points[new_ol_set[0]]
            for i in new_ol_set[1:]:
                new_points = list(new_points) + list(fil_points[i])
            ol_points.append(new_points)
        
            # Compute new boxes
            x_coords = np.array(new_points)[:,0]
            y_coords = np.array(new_points)[:,1]
            x_axis = [min(x_coords), max(x_coords)]
            y_axis = [min(y_coords), max(y_coords)]
            new_box = [y_axis, x_axis]

            ol_circles.append(circle_rect(new_box))
        

    gs_image = image

    ip_mask = np.zeros(gs_image.shape)

    # For each circle, paint it in the mask
    if verbose:
        print('6/6 | Masking sattelites')

    for circle in tqdm(ol_circles):
        cv2.circle(ip_mask, circle[0], math.ceil(circle[1]) + circle_mask_padding ,255,-1)

    ip_mask =  np.array(ip_mask, dtype = np.uint8)

    #mask_content = cv2.bitwise_and(gs_image, gs_image, mask = ip_mask)

    masked_image = cv2.bitwise_and(gs_image, gs_image, mask = cv2.bitwise_not(ip_mask))

    return masked_image


if __name__ == "__main__":
    # Run test
    # Import test 

    #path = "notebooks/data/20220722PM_ATTO2_T1_Split1.nd2"
    path = "notebooks/data/20220722PM_Calcein2_T1_Split1.nd2"
    #path = "notebooks/data/20220722PM_SulfoB1_T1_Split2.nd2"
    #path = "notebooks/data/20220722PM_CF405S_2_t1.nd2"
    #path = "notebooks/data/20220722PM_Calcein1_T1_Split2.nd2"
    
    test_image = nd2.imread(path)[-1]

    masked_image = clean_satellites(test_image)

    from matplotlib import pyplot as plt
    plt.imshow(test_image)
    plt.show()

    plt.imshow(masked_image)
    plt.show()
    
