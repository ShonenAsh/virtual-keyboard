import cv2
import numpy as np
import json
import sys
import pickle

def extract_region_coordinates(annotations_df):
    """
    Extract region coordinates from VIA annotations DataFrame
    
    Args:
        annotations_df: DataFrame with VIA annotations
        
    Returns:
        List of dictionaries with region_id and points (coordinates)
    """
    region_coordinates = {}
    for _, row in annotations_df.iterrows():
        try:
            shape_attr = json.loads(row['region_shape_attributes'])
            region_attr = json.loads(row['region_attributes'])
        except:
            print(f"Error with attribution file. Correct the attribution file and try again.")
            sys.exit(1)
        
        if 'x' in shape_attr and 'y' in shape_attr and 'width' in shape_attr and 'height' in shape_attr:
                x, y = shape_attr['x'], shape_attr['y']
                w, h = shape_attr['width'], shape_attr['height']
                points = np.array([(x, y), (x+w, y), (x+w, y+h), (x, y+h)], dtype=np.float32)
                region_coordinates[region_attr['name']]= points
    
    return region_coordinates

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
DETECTOR_PARAMS = cv2.aruco.DetectorParameters()
DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, DETECTOR_PARAMS)

def detectArucoMarkers(img):
    """
    Extract aruco markers from the image
    
    Args:
        img: video stream output
        
    Returns:
        List of keyboard endpoints (closest aruco marker edge detected to the keyboard)
    """
    markerCorners, markerIds, rejCandidates = DETECTOR.detectMarkers(img)
    if markerIds is not None:
        markerIds = np.squeeze(markerIds)
    
    if (type(markerIds)==np.ndarray) and markerIds.size == 4:
        marker_dict = {markerIds[i]:np.squeeze(markerCorners[i]) for i in range(4)}
        marker_mappings = {0:2,1:3,2:1,3:0}
        # cv2.aruco.drawDetectedMarkers(img, markerCorners, markerIds)
        for id in marker_dict:
            if id==0:
                top_left=marker_dict[id][marker_mappings[id]]
            elif id==1:
                top_right=marker_dict[id][marker_mappings[id]]
            elif id==2:
                bottom_left=marker_dict[id][marker_mappings[id]]
            elif id==3:
                bottom_right=marker_dict[id][marker_mappings[id]]
        keyboard = np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32)
        # for point in keyboard:
        #     cv2.circle(img, tuple(point.astype(int)), 5, (0, 255, 0), 2)
        return keyboard
    else:
        return None

def get_destination_points(src_points, multiplier=1.3):
    """return the size of the output image for perspective wrap"""
    # Unpack source points
    (tl, tr, bl, br) = src_points

    # Compute width (max of top and bottom width)
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    width = int(max(width_top, width_bottom)*multiplier)

    # Compute height (max of left and right height)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    height = int(max(height_left, height_right)*multiplier)

    # Define destination points in order: top-left, top-right, bottom-left, bottom-right
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [0, height - 1],
        [width - 1, height - 1]
    ], dtype=np.float32)

    return dst_points

def origin_translation(regions_coords_org,new_org):
    """origin translation whenever the origin is further away"""
    return {alphabet:regions_coords_org[alphabet]-new_org for alphabet in regions_coords_org}

def map_annotations(original_corners, target_corners, annotation_regions):
    """Enhanced homography calculation with error checking"""
    if len(original_corners) != 4 or len(target_corners) != 4:
        raise ValueError("Invalid number of corners for homography calculation")
    
    # Calculate homography matrix with RANSAC
    H, status = cv2.findHomography(np.float32(original_corners),
                                  np.float32(target_corners),
                                  cv2.RANSAC,
                                  5.0)
    
    if H is None or np.linalg.det(H) < 1e-6:
        raise ValueError("Homography matrix is invalid or degenerate")
    
    mapped_regions = {}
    for region_id, points in annotation_regions.items():
        # Transform points using perspective transformation
        points_reshaped = points.reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(points_reshaped, H)
        mapped_regions[region_id] = transformed_points.reshape(-1, 2)
    
    return mapped_regions

# Improved region detection using polygon testing
def point_in_region(regions, point):
    """Check if point is inside any region using polygon testing"""
    x, y = point
    for region, corners in regions.items():
        # Convert to contour format
        contour = np.array(corners, dtype=np.float32).reshape((-1, 1, 2))
        
        # Perform point-in-polygon test
        result = cv2.pointPolygonTest(contour, (float(x), float(y)), False)
        
        if result >= 0:  # Point is inside or on edge
            return region
    return None

def keyboard_map(frame):
    """Improved keyboard mapping with error handling"""
    keyboard_corners = detectArucoMarkers(frame)
    
    if keyboard_corners is not None and len(keyboard_corners) == 4:
        try:
            # Calculate destination points
            dst = get_destination_points(keyboard_corners)
            
            # Validate perspective matrix
            perspective_matrix = cv2.getPerspectiveTransform(keyboard_corners, dst, cv2.SOLVEPNP_ITERATIVE)
            if np.linalg.det(perspective_matrix) < 1e-6:
                raise ValueError("Invalid perspective matrix")
            
            # Generate keyboard snippet
            keyboard_snip = cv2.warpPerspective(frame, perspective_matrix, 
                                              (int(dst[3][0]), int(dst[3][1])))
            
            # Map annotations with error handling
            mapped_regions=map_annotations(corners_base,keyboard_corners,annotation_dict)
            mapped_regions_snip = map_annotations(corners_base, dst, annotation_dict)
            
            # Draw transformed regions on original frame
            for region_id, points in mapped_regions.items():
                pts = points.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
            
            return frame, perspective_matrix, keyboard_snip, mapped_regions_snip
            
        except Exception as e:
            print(f"Keyboard mapping error: {str(e)}")
            return frame, None, None, None
    
    return frame, None, None, None

with open("./keyboard_markers.pkl", 'rb') as f: 
    loaded_variables = pickle.load(f)
    annotation_dict = loaded_variables['annotation_dict']
    corners_base = loaded_variables['corners']
