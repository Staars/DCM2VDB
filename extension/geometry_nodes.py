"""Geometry nodes setup for volume to mesh conversion"""

import bpy
from .dicom_io import log
from .constants import GEONODES_DEFAULT_THRESHOLD, GEONODES_THRESHOLD_OFFSET

def create_volume_to_mesh_geonodes(vol_obj):
    """Create a Geometry Nodes modifier with Volume to Mesh setup"""
    
    try:
        log("Creating Geometry Nodes modifier...")
        
        # Add geometry nodes modifier
        mod = vol_obj.modifiers.new(name="VolumeToMesh", type='NODES')
        
        # Create new node group
        node_group = bpy.data.node_groups.new(name="CT_VolumeToMesh", type='GeometryNodeTree')
        mod.node_group = node_group
        
        # Create input/output nodes
        nodes = node_group.nodes
        links = node_group.links
        
        group_input = nodes.new('NodeGroupInput')
        group_input.location = (-600, 0)
        
        group_output = nodes.new('NodeGroupOutput')
        group_output.location = (600, 0)
        
        # Add Math node (Add) for threshold adjustment
        math_add = nodes.new('ShaderNodeMath')
        math_add.operation = 'ADD'
        math_add.location = (-300, 0)
        math_add.inputs[1].default_value = GEONODES_THRESHOLD_OFFSET
        
        # Add Volume to Mesh node
        vol_to_mesh = nodes.new('GeometryNodeVolumeToMesh')
        vol_to_mesh.location = (0, 0)
        vol_to_mesh.resolution_mode = 'GRID'
        
        # Add Set Material node
        set_material = nodes.new('GeometryNodeSetMaterial')
        set_material.location = (300, 0)
        
        # Get the CT_Mesh_Material
        mesh_material = bpy.data.materials.get("CT_Mesh_Material")
        if mesh_material:
            set_material.inputs['Material'].default_value = mesh_material
        
        # Create input sockets
        node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        node_group.interface.new_socket(name="Threshold", in_out='INPUT', socket_type='NodeSocketFloat')
        node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
        
        # Set default threshold value
        mod["Input_2_attribute_name"] = "density"
        mod["Input_2_use_attribute"] = 0
        mod["Input_2"] = GEONODES_DEFAULT_THRESHOLD
        
        # Connect nodes
        links.new(group_input.outputs[0], vol_to_mesh.inputs['Volume'])
        links.new(group_input.outputs[1], math_add.inputs[0])
        links.new(math_add.outputs[0], vol_to_mesh.inputs['Threshold'])
        links.new(vol_to_mesh.outputs['Mesh'], set_material.inputs['Geometry'])
        links.new(set_material.outputs['Geometry'], group_output.inputs[0])
        
        log(f"Created Geometry Nodes: Volume to Mesh with material (Threshold: {GEONODES_DEFAULT_THRESHOLD} HU + {GEONODES_THRESHOLD_OFFSET})")
        
    except Exception as e:
        log(f"ERROR creating Geometry Nodes: {e}")
        import traceback
        traceback.print_exc()
