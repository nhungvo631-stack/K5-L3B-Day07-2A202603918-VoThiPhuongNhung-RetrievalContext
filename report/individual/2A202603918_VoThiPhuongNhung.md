# Báo cáo Day 7 — 2A202603918 Võ Thị Phương Nhung

> **Trạng thái trung thực:** retrieval pipeline và giao diện đã chạy; phần agent trả lời bằng LLM chưa chạy vì chưa cấu hình `OPENAI_API_KEY` và `OPENAI_MODEL`. Không có nhãn grounded-answer, citation-support hoặc abstention nào được suy đoán.

## Dữ liệu và benchmark

- **Nguồn:** *Pass It On: Stories, Volume Two*, Foundation for a Better Life, PDF 58 trang; manifest chọn 32 trang có nội dung truyện. PDF trong workspace có quyền sử dụng giới hạn, không độc quyền, có thể thu hồi cho mục đích giáo dục/cá nhân phi thương mại; giữ thông báo bản quyền. Không đưa PDF vào Git.
- **Chunk:** trích text theo trang từ PDF; 174 chunk, 500 ký tự, overlap 50 ký tự. Mỗi chunk giữ `source_id`, URL nguồn, trang, `story_id` và ID ổn định.
- **Day 06:** metadata ghi Chroma Persistent 1.5.9, cosine, embedding `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 chiều. Database gốc tại `D:\Test\K5-L3B-Day06-VoThiPhuongNhung-2A202603918-DataIngestionEngineering\starter\results\chroma` lỗi `database disk image is malformed`; không sửa hoặc xóa database đó. Theo cho phép của người dùng, tạo index Chroma mới trong `starter/results/day7/chroma` từ PDF, cùng model và schema metadata. Cache thiếu cấu hình SentenceTransformers nên thư viện tạo mean-pooling mặc định; cần xác nhận với cấu hình ingest gốc trước khi xem index rebuild là tương đương bit-for-bit.
- **Benchmark:** 24 câu: dev 15 có đáp án + 2 thiếu evidence; test 5 có đáp án + 2 thiếu evidence. Evidence answerable được kiểm tra khớp chunk đúng nguồn/trang. Bốn câu thiếu evidence ở `data/day7_absence_questions.json` được đối chiếu với toàn bộ PDF và có lý do vắng mặt.
- **Phân tách dữ liệu:** `gold` chỉ được dùng bởi evaluator; retriever và controller nhận câu hỏi, filter do user cung cấp và context retrieval, không nhận đáp án chuẩn.
- **Thử thách nâng cao:** không thực hiện; tài liệu có 58 trang, ít hơn ngưỡng 100 trang.

## Retrieval và thử nghiệm

Định nghĩa: Hit@1 là evidence đúng ở vị trí đầu; Evidence Hit@3 là có ít nhất một evidence hợp lệ trong top-3; Context Evidence Rate là evidence đó có mặt trong context cuối. Mẫu số chỉ gồm câu answerable. Thời gian là median thời gian retrieval mỗi câu trên máy local; chi phí embedding API = 0 USD (embedding local). Chưa gọi LLM nên chi phí LLM chưa có.

| Cấu hình | Phương pháp, top-k, filter, context | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Median ms / chi phí |
|---|---|---|---|---:|---:|---:|---|
| Baseline | TF-IDF, k=3, budget=800 chars | Lexical baseline | dev | 10/15 (66.7%) | 14/15 (93.3%) | 10/15 (66.7%) | 0.77 ms / $0 |
| Phương pháp 2 | BM25, k=3, budget=800 chars | BM25 lexical ranking | dev | 12/15 (80.0%) | 14/15 (93.3%) | 12/15 (80.0%) | 0.67 ms / $0 |
| Phương pháp 3 | Dense Chroma, k=3, budget=800 chars | Embedding retrieval Day 06 | dev | 12/15 (80.0%) | 15/15 (100%) | 12/15 (80.0%) | 34.10 ms / $0 |
| Cải thiện 1 | Hybrid RRF, dense + BM25, k=3, RRF=60, budget=800 | Hợp nhất lexical và dense để nâng rank | dev | 13/15 (86.7%) | 15/15 (100%) | 13/15 (86.7%) | 34.61 ms / $0 |
| Thử nghiệm thêm | Hybrid RRF, k=5, RRF=20, budget=1200 | Tăng candidates và đổi hằng số RRF | dev | 13/15 (86.7%) | 15/15 (100%) | 13/15 (86.7%) | 30.69 ms / $0 |
| Cải thiện 2 / đã chọn | Dense Chroma, k=3, budget=1600 | Tăng context budget để không bỏ evidence ở rank 2–3 | dev | 12/15 (80.0%) | 15/15 (100%) | 15/15 (100%) | 38.85 ms / $0 |
| Cấu hình đã chọn | Dense Chroma, k=3, budget=1600 | Chốt theo Evidence Hit@3, sau đó Context Evidence Rate | test | 3/5 (60%) | 5/5 (100%) | 3/5 (60%) | 28.08 ms / $0 |
| Baseline | TF-IDF, k=3, budget=800 | So sánh cùng test | test | 4/5 (80%) | 5/5 (100%) | 4/5 (80%) | 1.25 ms / $0 |

### Phân tích lỗi và quyết định

- `q15` (“What endurance feat did Lawrence complete?”): TF-IDF và BM25 không đưa evidence vào top-3; từ khóa “endurance feat” khác cách diễn đạt “iron-distance triathlons”. Dense và hybrid tìm được evidence. Đây là lỗi lexical mismatch, không phải lỗi thiếu dữ liệu.
- Dense với budget 800 có ba trường hợp dev (`q01`, `q13`, `q14`) tìm thấy evidence trong top-3 nhưng không đưa vào context cuối vì các chunk trước chiếm ngân sách. Budget 1600 đưa evidence của cả 15/15 câu dev vào context, nên được chọn dù tăng median retrieval khoảng 4.75 ms so với dense budget 800.
- Hybrid RRF=60 cải thiện Hit@1 và Context Evidence Rate từ 12/15 lên 13/15 so với dense budget 800 nhưng không cải thiện thêm khi đổi sang RRF=20/k=5/budget=1200. Cấu hình này không được chọn vì context evidence thấp hơn dense budget 1600.
- Test cho thấy giới hạn ngoài dev: q17 và q20 có evidence ở rank 3 nhưng không vào context được dù budget 1600; q19 của baseline TF-IDF cũng có evidence ở rank 2 nhưng context chỉ giữ chunk rank 1. Đây là lỗi context selection; giữ nguyên kết quả test, không dùng test để chỉnh cấu hình.
- Không có rerank hoặc query rewrite. Filter `story_id` chỉ được áp dụng khi benchmark cung cấp filter tương ứng; metrics retrieval ở trên dùng filter đã lưu trong câu hỏi.

Trace và kết quả retrieval:

- `starter/results/day7/<config>/summary.json`
- `starter/results/day7/<config>/traces.json`
- Lệnh chạy có trong `starter/day7_pipeline.py` (`prepare`, `index`, `evaluate`); corpus và index local có thể tái tạo.

## Agent, context và giao diện

- **Agent:** `starter/run_agent.py` chạy batch; `starter/openai_compatible.py` gọi tool-calling API OpenAI-compatible; `starter/web_app.py` phục vụ giao diện local tại `http://127.0.0.1:8765`.
- **UI:** chọn Dev/Test, nhập câu hỏi, xem evidence/citation/trace. “Chỉ tìm bằng chứng” kiểm tra retrieval; “Hỏi agent thật” cần `OPENAI_BASE_URL`, `OPENAI_MODEL` và `OPENAI_API_KEY`. Điền trong `starter/.env` theo `starter/.env.example`, rồi khởi động lại server. Chưa có API/model được cấu hình trong lần chạy này, vì vậy chưa chạy agent thật hoặc gán điểm. Xem [AGENT_SETUP.md](../../starter/AGENT_SETUP.md).
- Agent giới hạn mặc định 3 lượt search, top-3, budget 1200 ký tự; context mỗi lượt được cập nhật lũy kế. Tool call, query, kết quả, context cuối, citations, số lượt, thời gian và lỗi được ghi vào trace. Lỗi không được đổi thành kết quả giả.
- **Chưa có số liệu agent:** Tool use, grounded answer 0/1/2, citation support và abstention để trống đến khi chạy model thật và người chấm audit thủ công. Các trace retrieval không được tính là câu trả lời của agent.
- **Cách chạy UI trên Windows:**

  ```powershell
  python starter/day7_pipeline.py prepare
  .venv\Scripts\python.exe starter\web_app.py --port 8765
  ```

  Sau đó mở `http://127.0.0.1:8765`. Đặt biến môi trường model/key trước khi khởi động server để bật nút “Hỏi agent thật”. API key không lưu trong mã hoặc trace.

## Giới hạn

- Index Chroma gốc của Day 06 không đọc được; index mới được tái tạo local từ PDF với cùng tên embedding model. Cần xác nhận cấu hình mean-pooling và so sánh embedding gốc nếu muốn khẳng định tương thích chính xác với vector index cũ.
- Test có 5 câu answerable nên tỷ lệ nhạy với một câu; kết quả retrieval không phải đánh giá câu trả lời. Chưa có model agent, nhãn thủ công, chi phí API hoặc kết quả abstention.
- PDF này không có đủ 100 trang và không được dùng cho bonus nâng cao.
