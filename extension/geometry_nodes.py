"""Geometry nodes setup for volume to mesh conversion"""

import bpy
from .utils import SimpleLogger

log = SimpleLogger()


def update_geometry_nodes_preset(context, preset_name):
    """Update geometry nodes HU thresholds when preset changes
    
    Args:
        context: Blender context
        preset_name: Name of new preset to apply
    """
    from .presets.material_presets import load_preset
    
    preset = load_preset(preset_name)
    if not preset or not preset.meshes:
        log.debug(f"No mesh definitions in preset {preset_name}")
        return
    
    # Create mapping of mesh name to HU threshold
    mesh_thresholds = {}
    for mesh_def in preset.meshes:
        mesh_name = mesh_def.get('name', '')
        mesh_threshold = mesh_def.get('threshold', 0)
        mesh_thresholds[mesh_name] = mesh_threshold
    
    log.info(f"Updating geometry nodes for preset: {preset_name}")
    
    # Find all objects with geometry nodes modifiers
    updated_count = 0
    for obj in bpy.data.objects:
        for mod in obj.modifiers:
            if mod.type == 'NODES' and mod.node_group:
                # Check if this is a tissue mesh modifier
                node_group = mod.node_group
                
                # Try to find the tissue name from node group name (e.g., "CT_Bone_Mesh")
                tissue_name = None
                for mesh_name in mesh_thresholds.keys():
                    if mesh_name in node_group.name:
                        tissue_name = mesh_name
                        break
                
                if not tissue_name:
                    continue
                
                new_threshold = mesh_thresholds[tissue_name]
                
                # Find the first math node (subtract node with HU threshold)
                for node in node_group.nodes:
                    if node.type == 'MATH' and node.operation == 'SUBTRACT':
                        if node.label == "HU Offset":
                            old_threshold = node.inputs[0].default_value
                            node.inputs[0].default_value = new_threshold
                            log.info(f"  Updated {obj.name}: {old_threshold} → {new_threshold} HU")
                            updated_count += 1
                            break
    
    if updated_count > 0:
        log.info(f"Updated {updated_count} geometry node(s) with new HU thresholds")
    else:
        log.debug("No geometry nodes found to update")


def create_tissue_mesh_geonodes(vol_obj, tissue_name, threshold_min, threshold_max, material=None):
    """Create a Geometry Nodes modifier to extract mesh at threshold (simplified - no boolean)"""
    
    try:
        # Get HU range for normalization
        scn = bpy.context.scene
        hu_min = -1024
        hu_max = 3071
        hu_range = hu_max - hu_min
        
        log.info(f"Creating Geometry Nodes for {tissue_name}...")
        log.info(f"  HU threshold: {threshold_min} (range: {hu_min} to {hu_max})")
        
        # Add geometry nodes modifier
        mod = vol_obj.modifiers.new(name=f"{tissue_name}_Mesh", type='NODES')
        
        # Create new node group
        node_group = bpy.data.node_groups.new(name=f"CT_{tissue_name}_Mesh", type='GeometryNodeTree')
        mod.node_group = node_group
        
        # Create input/output nodes
        nodes = node_group.nodes
        links = node_group.links
        
        group_input = nodes.new('NodeGroupInput')
        group_input.location = (-600, 0)
        
        group_output = nodes.new('NodeGroupOutput')
        group_output.location = (600, 0)
        
        # Math node 1: Subtract hu_min (HU → offset HU)
        math_subtract = nodes.new('ShaderNodeMath')
        math_subtract.operation = 'SUBTRACT'
        math_subtract.location = (-200, 0)
        math_subtract.label = "HU Offset"
        math_subtract.inputs[0].default_value = threshold_min  # HU threshold
        math_subtract.inputs[1].default_value = hu_min  # Subtract minimum
        
        # Math node 2: Divide by range (offset HU → normalized 0-1)
        math_divide = nodes.new('ShaderNodeMath')
        math_divide.operation = 'DIVIDE'
        math_divide.location = (0, 0)
        math_divide.label = "Normalize"
        math_divide.inputs[1].default_value = hu_range  # Divide by range
        
        # Volume to Mesh node
        vol_to_mesh = nodes.new('GeometryNodeVolumeToMesh')
        vol_to_mesh.location = (200, 0)
        
        # Set Material node (if provided)
        set_material = nodes.new('GeometryNodeSetMaterial')
        set_material.location = (400, 0)
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
        
        # Connect nodes: Input → Volume to Mesh → Math nodes → Set Material → Output
        links.new(group_input.outputs[0], vol_to_mesh.inputs['Volume'])
        
        # Connect math nodes for threshold calculation
        links.new(math_subtract.outputs[0], math_divide.inputs[0])  # Subtract → Divide
        links.new(math_divide.outputs[0], vol_to_mesh.inputs['Threshold'])  # Divide → Volume to Mesh
        
        # Connect mesh output
        links.new(vol_to_mesh.outputs['Mesh'], set_material.inputs['Geometry'])
        links.new(set_material.outputs['Geometry'], group_output.inputs[0])
        
        log.info(f"Created Geometry Nodes with math conversion: {tissue_name} (HU: {threshold_min})")
        
        return mod
        
    except Exception as e:
        log.info(f"ERROR creating Geometry Nodes for {tissue_name}: {e}")
        import traceback
        traceback.print_exc()
        return None
