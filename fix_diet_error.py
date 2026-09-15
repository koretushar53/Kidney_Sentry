#!/usr/bin/env python3
"""Fix the diet recommendation error handling to print actual exceptions."""

path = "main.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old_text = '''    except Exception as e:
        return "Diet recommendation unavailable: unable to fetch the recommendation."'''

new_text = '''    except Exception as e:
        error_msg = str(e)
        print(f"❌ Diet recommendation error: {error_msg}")
        return f"Diet recommendation unavailable: {error_msg}"'''

if old_text in content:
    content = content.replace(old_text, new_text)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ File updated successfully")
else:
    print("❌ Could not find the target text in the file")
