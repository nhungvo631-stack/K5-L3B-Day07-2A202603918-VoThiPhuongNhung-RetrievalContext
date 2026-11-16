# Cấu hình model thật cho Day 07

Agent dùng API chat tương thích OpenAI có hỗ trợ tool calling. Không gửi API key vào chat, issue, trace hoặc repository.

1. Tạo file cấu hình cục bộ:

   ```powershell
   Copy-Item starter\.env.example starter\.env
   notepad starter\.env
   ```

2. Điền `OPENAI_MODEL` và `OPENAI_API_KEY`. Nếu dùng endpoint khác, thay `OPENAI_BASE_URL`. Chỉ chọn endpoint/model mà bạn có quyền dùng và kiểm tra điều khoản dữ liệu trước khi gửi nội dung tài liệu ra ngoài.
3. Dừng server Day 07 cũ bằng `Ctrl+C`, sau đó khởi động lại:

   ```powershell
   .venv\Scripts\python.exe starter\web_app.py --port 8765
   ```

4. Mở `http://127.0.0.1:8765`, nhập hoặc chọn câu hỏi rồi bấm **Hỏi agent thật**. Trạng thái đầu trang sẽ hiện tên model khi đã nạp cấu hình.
5. Để benchmark agent, chạy CLI trên dev trước, audit câu trả lời/citation thủ công, rồi mới chạy test:

   ```powershell
   .venv\Scripts\python.exe starter\run_agent.py --split dev --method dense --output starter\results\day7\agent-dense-dev
   ```

`starter/.env` bị Git ignore; chỉ `.env.example` không chứa secret được theo dõi. Agent ghi model calls, tool calls, citations và lỗi vào trace, nhưng không ghi API key. Model chat có thể nhận context trích từ PDF; không dùng provider nếu bạn chưa xác nhận quyền xử lý dữ liệu.
