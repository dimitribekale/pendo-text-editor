# Pendo Text Editor Plan

**Phase 1: Foundational Text Editor**

1.  **Project Structuring:** I will organize the project with a dedicated `src` folder for the Python source code.
2.  **Core Application Window:** I will develop the main window for Pendo using Python's `tkinter` library, which will include a title and basic window controls.
3.  **Text Area:** A central text widget will be integrated to allow for typing and editing text.
4.  **Menu Bar:** A menu bar will be added, featuring a "File" menu with essential functionalities like:
    *   **New:** To clear the text area for a new document.
    *   **Open:** To load an existing text file.
    *   **Save:** To save the changes to the current file.
    *   **Save As:** To save the text as a new file.

**Phase 2: Predictive Text Engine**

1.  **N-gram Model:** I will implement a simple n-gram model (specifically, a trigram model) to predict the next word. This model will learn from the words you type.
2.  **Real-time Suggestions:** As you type, Pendo will analyze the last two words and suggest the most probable next word.
3.  **Suggestion Display:** The suggestion will appear as ghost text right after your cursor.
4.  **Accepting Suggestions:** You can easily accept a suggestion by pressing the `Tab` key.

**Phase 3: Advanced Features and UI/UX**

1.  **Edit Menu:** An "Edit" menu will be added with functionalities like "Cut," "Copy," "Paste," "Undo," and "Redo."
2.  **Enhanced UI:** The user interface will be improved with a status bar to show the current line and column number.
3.  **Search and Replace:** A "Search and Replace" feature will be implemented to quickly find and change text.

**Phase 4: UI/UX Improvement Suggestions**

### Core UI Enhancements
*   **Modern Styling:** Use the `ttk` themed widgets from `tkinter` to give the application a more modern and native look and feel on different operating systems.
*   **Icon Toolbar:** Add a toolbar with icons for frequently used actions like "New," "Open," "Save," "Cut," "Copy," and "Paste" for quicker access.
*   **Tabbed Interface:** Implement a tabbed document interface to allow users to open and edit multiple files in the same window.
*   **Line Numbers:** Display line numbers in a gutter on the left side of the text area.

### Feature-Specific UI Improvements
*   **Predictive Text:** Instead of just ghost text, display a small dropdown list of the top few word suggestions. Also, add an option in the menu or status bar to easily enable or disable this feature.
*   **Enhanced Status Bar:** Include more information in the status bar, such as word count, character count, and the current file's encoding (e.g., UTF-8).
*   **Context Menu:** Implement a right-click context menu in the text area for quick access to "Cut," "Copy," "Paste," and "Undo."

### Advanced Features
*   **Settings/Preferences:** Create a dedicated settings window where users can customize things like font size, color themes (e.g., light/dark mode), and predictive text behavior.
*   **Syntax Highlighting:** For a more powerful editor, consider adding syntax highlighting for common programming and markup languages.