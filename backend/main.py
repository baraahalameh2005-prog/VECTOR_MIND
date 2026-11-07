"""
main.py — entry point: Terminal Loop for asking questions.

Commands:  exit -> quit      new -> start a new topic (clears the conversation history)
"""

from rag_pipeline import ask, build_index


def print_result(result):
    print(f"\n{result['answer']}\n")
    if result["sources"]:
        print("المصادر:")
        for source in result["sources"]:
            print(f"  - {source['source']} | صفحة {source['page']}")
    print()


def main():
    print("جاري تجهيز قاعدة المعرفة...")
    build_index()
    print("\nالمرشد الأكاديمي الذكي — اكتب سؤالك.  (exit للخروج، new لموضوع جديد)\n")

    history = []
    while True:
        try:
            question = input("سؤالك: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() == "exit":
            break
        if question.lower() == "new":
            history = []
            print("تم بدء موضوع جديد.\n")
            continue

        try:
            result = ask(question, history)
        except RuntimeError as error:
            print(f"\n[خطأ] {error}\n")
            continue

        print_result(result)
        history.append({"question": question, "answer": result["answer"]})


if __name__ == "__main__":
    main()
