"""Geometry nodes setup for volume to mesh conversion"""

import bpy
from .dicom_io import log

def create_tissue_mesh_geonodes(vol_obj, tissue_name, threshold_min, threshold_max, material=None):
    """Create a Geometry Nodes modifier to extract mesh at threshold (simplified - no boolean)"""
    
    try:
        # Convert HU threshold to normalized 0-1 range
        scn = bpy.context.scene
        hu_min = scn.dicom_volume_hu_min
        hu_max = scn.dicom_volume_hu_max
        threshold_normalized = (threshold_min - hu_min) / (hu_max - hu_min)
        
        log(f"Creating Geometry Nodes for {tissue_name}...")
        log(f"  HU threshold: {threshold_min} -> normalized: {threshold_normalized:.6f}")
        
        # Add geometry nodes modifier
        mod = vol_obj.modifiers.new(name=f"{tissue_name}_Mesh", type='NODES')
        
        # Create new node group
        node_group = bpy.data.node_groups.new(name=f"CT_{tissue_name}_Mesh", type='GeometryNodeTree')
        mod.node_group = node_group
        
        # Create input/output nodes
        nodes = node_group.nodes
        links = node_group.links
        
        group_input = nodes.new('NodeGroupInput')
        group_input.location = (-400, 0)
        
        group_output = nodes.new('NodeGroupOutput')
        group_output.location = (400, 0)
        
        # Volume to Mesh node (simple - just lower threshold)
        vol_to_mesh = nodes.new('GeometryNodeVolumeToMesh')
        vol_to_mesh.location = (0, 0)
        # resolution_mode removed in Blender 5.0
        vol_to_mesh.inputs['Threshold'].default_value = threshold_normalized
        
        # Set Material node (if provided)
        set_material = nodes.new('GeometryNodeSetMaterial')
        set_material.location = (200, 0)
        if material:
            set_material.inputs['Material'].default_value = material
        
        # Create sockets - try different API for Blender 5.0
        try:
            # Blender 4.0+ API
            node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
            node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        except:
            # Fallback for older/newer versions
            try:
                output_socket = node_group.interface.items_tree.new('NodeSocketGeometry', 'Geometry')
                output_socket.in_out = 'OUTPUT'
                input_socket = node_group.interface.items_tree.new('NodeSocketGeometry', 'Geometry')
                input_socket.in_out = 'INPUT'
            except:
                # Last resort - use outputs/inputs directly
                node_group.outputs.new('NodeSocketGeometry', 'Geometry')
                node_group.inputs.new('NodeSocketGeometry', 'Geometry')
        
        # Connect nodes (simple path - no boolean)
        links.new(group_input.outputs[0], vol_to_mesh.inputs['Volume'])
        links.new(vol_to_mesh.outputs['Mesh'], set_material.inputs['Geometry'])
        links.new(set_material.outputs['Geometry'], group_output.inputs[0])
        
        log(f"Created simple Geometry Nodes: {tissue_name} (threshold: {threshold_min} HU)")
        
        return mod
        
    except Exception as e:
        log(f"ERROR creating Geometry Nodes for {tissue_name}: {e}")
        import traceback
        traceback.print_exc()
        return None
