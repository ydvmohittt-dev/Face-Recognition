import os
import shutil
from pathlib import Path


def Remove_file(file_path: str = None):
    """
    Remove all images for a specific person in the Attendance_data subfolder.
    Prompts for confirmation to avoid accidental deletion.

    Args:
        file_path (str): Parent directory (default: 'Attendance_data').
    """
    if file_path is None:
        file_path = "Attendance_data"

    # Validate parent directory
    if not os.path.exists(file_path):
        print(f"❌ Error: Directory '{file_path}' does not exist.")
        return

    # List all available persons
    try:
        available_persons = [d for d in os.listdir(file_path)
                             if os.path.isdir(os.path.join(file_path, d)) and not d.startswith('.')]

        if not available_persons:
            print(f"⚠️ No person folders found in '{file_path}'.")
            return

        print(f"\n📁 Available persons in database ({len(available_persons)}):")
        for i, person in enumerate(sorted(available_persons), 1):
            person_dir = os.path.join(file_path, person)
            img_count = len([f for f in os.listdir(person_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"  {i}. {person} ({img_count} images)")
        print()

    except Exception as e:
        print(f"❌ Error listing persons: {e}")
        return

    # Get person name to delete
    Remove_file_name = input(
        "Please Enter the name to delete images for: ").strip().upper()

    if not Remove_file_name:
        print("❌ Error: Name cannot be empty.")
        return

    person_dir = os.path.join(file_path, Remove_file_name)

    # Check if person's folder exists
    if not os.path.isdir(person_dir):
        print(
            f"❌ Error: No folder found for '{Remove_file_name}' in '{file_path}'.")
        print(f"💡 Tip: Available persons are listed above (case-sensitive).")
        return

    # Count images in the folder
    try:
        image_files = [f for f in os.listdir(person_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')]

        if not image_files:
            print(
                f"⚠️ No images found in '{person_dir}'. Removing empty folder...")
            shutil.rmtree(person_dir)
            print(f"✅ Removed empty folder for '{Remove_file_name}'.")
            return

    except Exception as e:
        print(f"❌ Error reading directory '{person_dir}': {e}")
        return

    # Display deletion summary
    print(f"\n⚠️  DELETION SUMMARY:")
    print(f"  • Person: {Remove_file_name}")
    print(f"  • Location: {person_dir}")
    print(f"  • Images to delete: {len(image_files)}")
    print(f"  • Sample files: {', '.join(image_files[:3])}")
    if len(image_files) > 3:
        print(f"    ... and {len(image_files) - 3} more")

    # Confirm deletion with double-check
    confirm = input(
        "\n⚠️  Type 'DELETE' to confirm deletion (or anything else to cancel): ").strip()

    if confirm != 'DELETE':
        print("✅ Deletion canceled. No files were removed.")
        return

    # Final confirmation
    final_confirm = input(
        f"⚠️  Final confirmation: Delete all {len(image_files)} images for '{Remove_file_name}'? (y/n): ").strip().lower()

    if final_confirm != 'y':
        print("✅ Deletion canceled. No files were removed.")
        return

    # Delete the entire folder
    try:
        shutil.rmtree(person_dir)
        print(
            f"\n✅ Successfully deleted all {len(image_files)} images and folder for '{Remove_file_name}'.")
        print(f"💡 To re-add this person, run the capture script again.")

    except PermissionError:
        print(f"❌ Permission denied: Cannot delete '{person_dir}'.")
        print("💡 Try running the script with administrator/sudo privileges.")
    except Exception as e:
        print(f"❌ Error deleting '{person_dir}': {e}")


def batch_remove(file_path: str = None):
    """
    Remove multiple persons at once from the attendance database.

    Args:
        file_path (str): Parent directory (default: 'Attendance_data').
    """
    if file_path is None:
        file_path = "Attendance_data"

    if not os.path.exists(file_path):
        print(f"❌ Error: Directory '{file_path}' does not exist.")
        return

    # List all available persons
    try:
        available_persons = [d for d in os.listdir(file_path)
                             if os.path.isdir(os.path.join(file_path, d)) and not d.startswith('.')]

        if not available_persons:
            print(f"⚠️ No person folders found in '{file_path}'.")
            return

        print(f"\n📁 Available persons in database ({len(available_persons)}):")
        for i, person in enumerate(sorted(available_persons), 1):
            person_dir = os.path.join(file_path, person)
            img_count = len([f for f in os.listdir(person_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"  {i}. {person} ({img_count} images)")
        print()

    except Exception as e:
        print(f"❌ Error listing persons: {e}")
        return

    # Get names to delete
    names_input = input(
        "Enter names to delete (comma-separated): ").strip().upper()
    names_to_delete = [name.strip()
                       for name in names_input.split(',') if name.strip()]

    if not names_to_delete:
        print("❌ No names provided.")
        return

    # Validate names
    valid_names = []
    invalid_names = []

    for name in names_to_delete:
        person_dir = os.path.join(file_path, name)
        if os.path.isdir(person_dir):
            valid_names.append(name)
        else:
            invalid_names.append(name)

    if invalid_names:
        print(f"\n⚠️  Invalid names (not found): {', '.join(invalid_names)}")

    if not valid_names:
        print("❌ No valid names to delete.")
        return

    # Show summary
    print(f"\n⚠️  BATCH DELETION SUMMARY:")
    print(f"  • Persons to delete: {len(valid_names)}")
    total_images = 0
    for name in valid_names:
        person_dir = os.path.join(file_path, name)
        img_count = len([f for f in os.listdir(person_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        total_images += img_count
        print(f"    - {name}: {img_count} images")

    print(f"  • Total images to delete: {total_images}")

    # Confirm deletion
    confirm = input(
        "\n⚠️  Type 'DELETE ALL' to confirm batch deletion: ").strip()

    if confirm != 'DELETE ALL':
        print("✅ Batch deletion canceled.")
        return

    # Delete folders
    deleted_count = 0
    failed_count = 0

    for name in valid_names:
        person_dir = os.path.join(file_path, name)
        try:
            shutil.rmtree(person_dir)
            deleted_count += 1
            print(f"✅ Deleted: {name}")
        except Exception as e:
            failed_count += 1
            print(f"❌ Failed to delete {name}: {e}")

    print(f"\n📊 Batch deletion complete:")
    print(f"  • Successfully deleted: {deleted_count}")
    print(f"  • Failed: {failed_count}")


if __name__ == "__main__":
    print("=== Attendance Data Management ===")
    print("1. Remove single person")
    print("2. Remove multiple persons (batch)")
    print("3. Exit")

    choice = input("\nSelect option (1-3): ").strip()

    if choice == '1':
        Remove_file()
    elif choice == '2':
        batch_remove()
    elif choice == '3':
        print("✅ Exiting...")
    else:
        print("❌ Invalid choice. Exiting...")
