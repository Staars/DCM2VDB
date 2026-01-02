"""UI utility functions for Blender area and space manipulation"""

import bpy
from typing import List, Optional, Tuple


def find_image_editor_spaces(context) -> List[Tuple]:
    """
    Find all Image Editor spaces in the current window manager.
    
    Args:
        context: Blender context
        
    Returns:
        List of tuples: [(area, space), ...]
    """
    spaces = []
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                for space in area.spaces:
                    if space.type == 'IMAGE_EDITOR':
                        spaces.append((area, space))
    return spaces


def set_image_in_all_editors(context, image, clear_first: bool = False) -> bool:
    """
    Set an image as active in all Image Editor spaces.
    
    Args:
        context: Blender context
        image: bpy.types.Image to set (or None to clear)
        clear_first: If True, clear image before setting (forces refresh)
        
    Returns:
        True if at least one Image Editor was found, False otherwise
    """
    spaces = find_image_editor_spaces(context)
    
    if not spaces:
        return False
    
    for area, space in spaces:
        if clear_first:
            space.image = None
        space.image = image
        area.tag_redraw()
    
    return True


def clear_image_from_all_editors(context, image) -> int:
    """
    Clear a specific image from all Image Editor spaces.
    
    Args:
        context: Blender context
        image: bpy.types.Image to clear
        
    Returns:
        Number of editors where image was cleared
    """
    count = 0
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                for space in area.spaces:
                    if space.type == 'IMAGE_EDITOR' and space.image == image:
                        space.image = None
                        count += 1
    return count


def has_image_editor(context) -> bool:
    """
    Check if any Image Editor area exists.
    
    Args:
        context: Blender context
        
    Returns:
        True if at least one Image Editor exists
    """
    return len(find_image_editor_spaces(context)) > 0


def get_active_image_editor_space(context) -> Optional[Tuple]:
    """
    Get the active Image Editor space (if context is in Image Editor).
    
    Args:
        context: Blender context
        
    Returns:
        Tuple of (area, space) if in Image Editor, None otherwise
    """
    if context.area and context.area.type == 'IMAGE_EDITOR':
        space = context.space_data
        if space and space.type == 'IMAGE_EDITOR':
            return (context.area, space)
    return None


def refresh_all_image_editors(context) -> int:
    """
    Force redraw of all Image Editor areas.
    
    Args:
        context: Blender context
        
    Returns:
        Number of areas refreshed
    """
    count = 0
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                area.tag_redraw()
                count += 1
    return count


def refresh_all_3d_views(context) -> int:
    """
    Force redraw of all 3D View areas.
    
    Args:
        context: Blender context
        
    Returns:
        Number of areas refreshed
    """
    count = 0
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
                count += 1
    return count
