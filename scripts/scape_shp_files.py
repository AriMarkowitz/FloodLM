"""
scrape_shp_files.py
===================
Scrape node and link shapefiles into pandas DataFrames.

Two entry-point functions:
  - process_node_shapefiles(paths)  -> dict[label, DataFrame]
  - process_link_shapefiles(paths)  -> dict[label, DataFrame]

Each function reads every attribute field in the shapefile and also adds
derived geometric columns (x/y for nodes; length, start/end coords for links).
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiLineString


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _line_endpoints(geom):
    """Return (x_start, y_start, x_end, y_end) for a line geometry."""
    if geom is None or geom.is_empty:
        return np.nan, np.nan, np.nan, np.nan
    coords = list(geom.geoms[0].coords if isinstance(geom, MultiLineString) else geom.coords)
    return coords[0][0], coords[0][1], coords[-1][0], coords[-1][1]


# ---------------------------------------------------------------------------
# Node shapefiles
# ---------------------------------------------------------------------------

def process_node_shapefile(path: str):
    """
    Read one or more node shapefiles and return all their fields as DataFrames.

    Geometry columns added automatically:
      x, y  — centroid coordinates (from Point geometry, or centroid of polygon/line)

    Parameters
    ----------
    path : str path to a .shp file
            e.g. "/data/Nodes_2D.shp"

    Returns
    -------
    a DataFrame
    """
    
    print(f"[nodes] Reading {path}")
    gdf = gpd.read_file(path)
    print(f"        {len(gdf)} rows | CRS: {gdf.crs} | geom: {gdf.geom_type.unique()}")

    # Use centroid so x/y always exist regardless of geometry type
    centroids = gdf.geometry.centroid
    df = gdf.drop(columns="geometry").copy()
    df["x"] = centroids.x.values
    df["y"] = centroids.y.values

    print(f"        columns: {list(df.columns)}\n")

    return df


# ---------------------------------------------------------------------------
# Link / edge shapefiles
# ---------------------------------------------------------------------------

def process_link_shapefile(path: str):
    """
    Read one or more link/edge shapefiles and return all their fields as DataFrames.

    Geometry columns added automatically:
      length             — total length of the line in CRS units
      x_start, y_start  — coordinates of the first vertex
      x_end,   y_end    — coordinates of the last vertex

    Parameters
    ----------
    path : str path to a .shp file
            e.g. "/data/Node1D_to_Node2D_Links.shp"

    Returns
    -------
    a DataFrame
    """


    print(f"[links] Reading {path}")
    gdf = gpd.read_file(path)
    print(f"        {len(gdf)} rows | CRS: {gdf.crs} | geom: {gdf.geom_type.unique()}")

    df = gdf.drop(columns="geometry").copy()

    # Geometric derived columns
    df["length"] = gdf.geometry.length.values

    endpoints = gdf.geometry.apply(_line_endpoints)
    df["x_start"] = [ep[0] for ep in endpoints]
    df["y_start"] = [ep[1] for ep in endpoints]
    df["x_end"]   = [ep[2] for ep in endpoints]
    df["y_end"]   = [ep[3] for ep in endpoints]

    print(f"        columns: {list(df.columns)}\n")


    return df



# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

MODEL2_PATH = "/Users/rishi/urban flooding/Models/Model_2"

######### NODES 1D ######### 
nodes_1d = process_node_shapefile(f"{MODEL2_PATH}/shapefiles/Nodes_1D.shp")
nodes_1d.rename(columns={'FID': 'node_idx'}, inplace=True)
# nodes_1d.columns == ['FID', 'Name', 'SystemName', 'NodeType', 'NodeStatus', 'ConnecUSDS', 'InvertElev', 'BaseArea', 'TrrainElev', 'ElvOvrride', 'Depth', 'DrpInltE', 'DrpInltWL', 'DrpInltWC', 'DrpInltA', 'DrpInltOC', 'TotConnec', 'X', 'Y', 'x', 'y']

# Create one-hot encoding for NodeType
node_type_dummies = pd.get_dummies(nodes_1d['NodeType'], prefix='NodeType')
nodes_1d = pd.concat([nodes_1d, node_type_dummies], axis=1)
# Create binary column for NodeStatus containing 'with droop inlet'
nodes_1d['has_drop_inlet'] = nodes_1d['NodeStatus'].str.contains('with drop inlet', case=False, na=False).astype(int)
# Split ConnecUSDS into separate columns
nodes_1d[['ConnectUS', 'ConnectDS']] = nodes_1d['ConnecUSDS'].str.split(':', expand=True).astype(int)

# Update columns to save
columns_to_save = ['node_idx', 'NodeType_Boundary', 'NodeType_Junction', 'NodeType_Start', 'has_drop_inlet', 'ConnectUS', 'ConnectDS']
# # Load existing CSV and perform left join
static_file = f"{MODEL2_PATH}/train/1d_nodes_static.csv"
existing_df = pd.read_csv(static_file)
print(existing_df.columns, nodes_1d[columns_to_save].columns)
merged_df = existing_df.merge(nodes_1d[columns_to_save], on="node_idx", how="left")
merged_df.to_csv(f"{MODEL2_PATH}/train/1d_nodes_static_expanded.csv", index=False)
print(f"Saved merged data with node_idx to 1d_nodes_static_expanded.csv")

######## LINKS 1D #########
links_1d = process_link_shapefile(f"{MODEL2_PATH}/shapefiles/Links_1D.shp")
links_1d.rename(columns={'FID': 'edge_idx'}, inplace=True)
# print(links_1d.columns)
# lots of interesting stuff!
# ['FID', 'Count', 'Name', 'SystemName', 'USNode', 'DSNode', 'ModelingAp', 'Length', 'MeshCellLe', 'Shape', 'Rise', 'Span', "Manning'sn", 'USOffset', 'DSOffset', 'USElevatio', 'DSElevatio', 'Slope', 'USEnLoss', 'DSExLoss', 'USBFLoss', 'DSBFLoss', 'DSGateType', 'MajorGroup', 'MinorGroup', 'length', 'x_start', 'y_start', 'x_end', 'y_end']

# Save specific columns to CSV
columns_to_save = ['edge_idx', 'USEnLoss', 'DSExLoss', 'USBFLoss', 'DSBFLoss']

# # Load existing CSV and perform left join
static_file = f"{MODEL2_PATH}/train/1d_edges_static.csv"
existing_df = pd.read_csv(static_file)
# print(existing_df.columns, links_1d[columns_to_save].columns)
merged_df = existing_df.merge(links_1d[columns_to_save], on="edge_idx", how="left")
merged_df.to_csv(f"{MODEL2_PATH}/train/1d_edges_static_expanded.csv", index=False)
print(f"Saved merged data with edge_idx to 1d_edges_static_expanded.csv")

#---------------

######## NODES 2D #########
# nodes_2d = process_node_shapefile(f"{MODEL2_PATH}/shapefiles/Nodes_2D.shp")
# print(nodes_2d.columns)
# all expected information that I think we are already using
#['FID', 'min_ele', 'manning_n', 'area', 'Centre_ele', 'Type', 'x', 'y']

######## LINKS 2D #########
# links_2d = process_link_shapefile(f"{MODEL2_PATH}/shapefiles/Links_2D.shp")
# print(links_2d.columns)
# frm_node_E and to_node_E are probably not that interesting, information alr in node data
# ['FID', 'from_node', 'to_node', 'length', 'frm_node_E', 'to_node_E', 'slope', 'x_start', 'y_start', 'x_end', 'y_end']

######## LINKS 1D2D #########
# links_1d2d = process_link_shapefile(f"{MODEL2_PATH}/shapefiles/Node1D_to_Node2D_Links.shp")
# print(links_1d2d.columns)
# Nothing interesting, just node ends and length
# ['node_1d', 'node_2d', 'length', 'x_start', 'y_start', 'x_end', 'y_end']

