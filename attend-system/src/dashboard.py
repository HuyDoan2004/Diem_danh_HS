"""
Streamlit Dashboard for Attendance System
Giao diện web để quản lý và xem báo cáo điểm danh
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from database import DatabaseHandler


# Page config
st.set_page_config(
    page_title="Attendance System Dashboard",
    page_icon="[SYSTEM]",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_database():
    """Get database handler (cached)"""
    return DatabaseHandler()


def format_time(dt):
    """Format datetime to HH:MM"""
    if dt:
        return dt.strftime('%H:%M:%S')
    return 'N/A'


def show_overview(db):
    """Trang tổng quan"""
    st.markdown('<div class="main-header">[SYSTEM] HỆ THỐNG ĐIỂM DANH HỌC SINH</div>', unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    students = db.get_all_students()
    
    with col1:
        st.metric("Tổng số học sinh", len(students))
    
    # Get today's sessions
    today = datetime.now().date()
    today_sessions = db.get_session_by_date(today)
    
    with col2:
        st.metric("Buổi học hôm nay", len(today_sessions))
    
    # Count today's attendances
    today_attendances = []
    for session in today_sessions:
        today_attendances.extend(db.get_attendance_by_session(session.id))
    
    with col3:
        present_count = len([a for a in today_attendances if a.status in ['present', 'late']])
        st.metric("Đã điểm danh", present_count)
    
    with col4:
        if today_sessions:
            rate = (present_count / len(students)) * 100 if students else 0
            st.metric("Tỷ lệ tham gia", f"{rate:.1f}%")
        else:
            st.metric("Tỷ lệ tham gia", "N/A")
    
    st.divider()
    
    # Today's schedule
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("[SCHEDULE] Lịch học hôm nay")
        
        day_of_week = datetime.now().strftime('%A').lower()
        schedules = db.get_schedule_for_day(day_of_week)
        
        if schedules:
            schedule_data = []
            for s in schedules:
                schedule_data.append({
                    'Môn học': s.subject,
                    'Thời gian': f"{s.start_time} - {s.end_time}",
                    'Phòng': s.room,
                    'Giáo viên': s.teacher
                })
            
            df_schedule = pd.DataFrame(schedule_data)
            st.dataframe(df_schedule, use_container_width=True, hide_index=True)
        else:
            st.info("Không có lịch học hôm nay")
    
    with col2:
        st.subheader("[STATS] Điểm danh hôm nay")
        
        if today_attendances:
            status_counts = pd.Series([a.status for a in today_attendances]).value_counts()
            
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Trạng thái điểm danh",
                color_discrete_map={
                    'present': '#00CC96',
                    'late': '#FFA15A',
                    'absent': '#EF553B'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Chưa có dữ liệu điểm danh hôm nay")


def show_students(db):
    """Trang quản lý học sinh"""
    st.header(" Quản lý học sinh")
    
    tab1, tab2, tab3 = st.tabs(["Danh sách", "Thêm mới", "Import Excel"])
    
    with tab1:
        students = db.get_all_students()
        
        if students:
            student_data = []
            for s in students:
                stats = db.get_attendance_statistics(s.id)
                student_data.append({
                    'MSSV': s.student_id,
                    'Họ tên': s.name,
                    'Lớp': s.class_name or 'N/A',
                    'Số buổi': stats['total'],
                    'Có mặt': stats['present'],
                    'Đi trễ': stats['late'],
                    'Vắng': stats['absent'],
                    'Tỷ lệ (%)': f"{stats['attendance_rate']:.1f}"
                })
            
            df = pd.DataFrame(student_data)
            
            # Search
            search = st.text_input(" Tìm kiếm học sinh", "")
            if search:
                df = df[df.apply(lambda row: search.lower() in row.astype(str).str.lower().to_string(), axis=1)]
            
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Download button
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                " Tải xuống CSV",
                csv,
                "students.csv",
                "text/csv",
                key='download-csv'
            )
        else:
            st.info("Chưa có học sinh nào trong hệ thống")
    
    with tab2:
        with st.form("add_student"):
            st.subheader("Thêm học sinh mới")
            
            col1, col2 = st.columns(2)
            
            with col1:
                student_id = st.text_input("MSSV *")
                name = st.text_input("Họ tên *")
            
            with col2:
                class_name = st.text_input("Lớp")
                email = st.text_input("Email")
            
            phone = st.text_input("Số điện thoại")
            
            submitted = st.form_submit_button(" Thêm học sinh")
            
            if submitted:
                if not student_id or not name:
                    st.error("Vui lòng điền MSSV và Họ tên!")
                else:
                    result = db.add_student(
                        student_id=student_id,
                        name=name,
                        class_name=class_name if class_name else None,
                        email=email if email else None,
                        phone=phone if phone else None
                    )
                    
                    if result:
                        st.success(f"[SUCCESS] Đã thêm học sinh: {name}")
                        st.rerun()
                    else:
                        st.error("Lỗi khi thêm học sinh (có thể MSSV đã tồn tại)")
    
    with tab3:
        st.subheader(" Import danh sách từ Excel")
        
        # Download template
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Bước 1: Tải template Excel**")
            template_data = {
                'student_id': ['SV001', 'SV002', 'SV003'],
                'name': ['Nguyen Van A', 'Tran Thi B', 'Le Van C'],
                'class_name': ['CNTT01', 'CNTT01', 'CNTT02'],
                'email': ['nva@example.com', 'ttb@example.com', 'lvc@example.com'],
                'phone': ['0901234567', '0902345678', '0903456789']
            }
            template_df = pd.DataFrame(template_data)
            
            # Convert to Excel
            from io import BytesIO
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                template_df.to_excel(writer, index=False, sheet_name='Students')
            
            st.download_button(
                " Tải template Excel",
                buffer.getvalue(),
                "student_template.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            st.info("""
            **Cột bắt buộc:**
            - `student_id`: Mã sinh viên (duy nhất)
            - `name`: Họ tên (dùng làm tên folder)
            
            **Cột tùy chọn:**
            - `class_name`: Lớp
            - `email`: Email
            - `phone`: Số điện thoại
            """)
        
        with col2:
            st.markdown("**Bước 2: Upload file Excel**")
            
            uploaded_file = st.file_uploader(
                "Chọn file Excel",
                type=['xlsx', 'xls', 'csv'],
                help="File phải có cột student_id và name"
            )
            
            create_folders = st.checkbox(
                "Tự động tạo folder cho từng học sinh",
                value=True,
                help="Tạo folder trong data/enroll/ để thu thập dữ liệu"
            )
            
            if uploaded_file:
                try:
                    # Import module
                    from import_students import StudentImporter
                    
                    # Save uploaded file temporarily
                    temp_path = Path("temp_upload.xlsx")
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Preview data
                    if uploaded_file.name.endswith('.csv'):
                        preview_df = pd.read_csv(temp_path)
                    else:
                        preview_df = pd.read_excel(temp_path)
                    
                    st.markdown("**Preview dữ liệu:**")
                    st.dataframe(preview_df.head(10), use_container_width=True)
                    
                    st.info(f"[STATS] Tổng số học sinh: {len(preview_df)}")
                    
                    # Import button
                    if st.button(" Import vào hệ thống", type="primary"):
                        with st.spinner("Đang import..."):
                            importer = StudentImporter(db)
                            result = importer.import_students(
                                temp_path,
                                create_folders=create_folders
                            )
                        
                        # Clean up
                        temp_path.unlink(missing_ok=True)
                        
                        # Show results
                        if result['success']:
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("[SUCCESS] Imported", result['imported'])
                            with col_b:
                                st.metric("[SKIP] Skipped", result['skipped'])
                            with col_c:
                                st.metric("[FAIL] Errors", len(result['errors']))
                            
                            if result['imported'] > 0:
                                st.success(f"[SUCCESS] Import thành công {result['imported']} học sinh!")
                                
                                if create_folders:
                                    st.info("[FOLDER] Folder đã được tạo trong data/enroll/")
                            
                            if result['errors']:
                                with st.expander("[ERROR] Chi tiết lỗi"):
                                    for error in result['errors']:
                                        st.error(error)
                            
                            # Refresh page
                            if result['imported'] > 0:
                                st.rerun()
                        else:
                            st.error(f"[FAIL] Import thất bại: {result['message']}")
                
                except Exception as e:
                    st.error(f"Lỗi: {str(e)}")
                    temp_path.unlink(missing_ok=True)


def show_attendance(db):
    """Trang theo dõi điểm danh"""
    st.header("[LIST] Theo dõi điểm danh")
    
    # Date selector
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_date = st.date_input("Chọn ngày", datetime.now().date())
    
    # Get sessions for selected date
    sessions = db.get_session_by_date(selected_date)
    
    if not sessions:
        st.info(f"Không có buổi học nào vào ngày {selected_date}")
        return
    
    # Session selector
    with col2:
        session_options = [f"{s.schedule.subject} ({s.schedule.start_time}-{s.schedule.end_time})" for s in sessions]
        selected_session_idx = st.selectbox("Chọn buổi học", range(len(sessions)), format_func=lambda x: session_options[x])
    
    selected_session = sessions[selected_session_idx]
    
    st.divider()
    
    # Session info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Môn học", selected_session.schedule.subject)
    with col2:
        st.metric("Phòng", selected_session.schedule.room or 'N/A')
    with col3:
        st.metric("Giáo viên", selected_session.schedule.teacher or 'N/A')
    with col4:
        st.metric("Trạng thái", selected_session.status)
    
    st.divider()
    
    # Attendance list
    attendances = db.get_attendance_by_session(selected_session.id)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Danh sách điểm danh")
        
        if attendances:
            attendance_data = []
            for att in attendances:
                attendance_data.append({
                    'MSSV': att.student.student_id,
                    'Họ tên': att.student.name,
                    'Lớp': att.student.class_name or 'N/A',
                    'Giờ vào': format_time(att.check_in_time),
                    'Giờ ra': format_time(att.check_out_time),
                    'Trạng thái': att.status.upper(),
                    'Độ tin cậy': f"{att.confidence:.2f}" if att.confidence else 'N/A'
                })
            
            df = pd.DataFrame(attendance_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Download
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                " Tải xuống",
                csv,
                f"attendance_{selected_date}_{selected_session.schedule.subject}.csv",
                "text/csv"
            )
        else:
            st.info("Chưa có học sinh nào điểm danh")
    
    with col2:
        st.subheader("Thống kê")
        
        total_students = len(db.get_all_students())
        present_count = len([a for a in attendances if a.status == 'present'])
        late_count = len([a for a in attendances if a.status == 'late'])
        absent_count = total_students - present_count - late_count
        
        st.metric("Tổng số", total_students)
        st.metric("Có mặt", present_count, delta=None)
        st.metric("Đi trễ", late_count, delta=None)
        st.metric("Vắng", absent_count, delta=None)
        
        if total_students > 0:
            rate = ((present_count + late_count) / total_students) * 100
            st.metric("Tỷ lệ tham gia", f"{rate:.1f}%")


def show_reports(db):
    """Trang báo cáo"""
    st.header("[STATS] Báo cáo & Thống kê")
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Từ ngày", datetime.now().date() - timedelta(days=30))
    with col2:
        end_date = st.date_input("Đến ngày", datetime.now().date())
    
    st.divider()
    
    # Student selection
    students = db.get_all_students()
    student_names = [s.name for s in students]
    
    if not student_names:
        st.warning("Chưa có học sinh trong hệ thống")
        return
    
    selected_student_name = st.selectbox("Chọn học sinh", student_names)
    
    selected_student = db.get_student(name=selected_student_name)
    
    if not selected_student:
        st.warning("Không tìm thấy học sinh")
        return
    
    # Get attendance history
    history = db.get_student_attendance_history(
        selected_student.id,
        start_date=datetime.combine(start_date, datetime.min.time()),
        end_date=datetime.combine(end_date, datetime.max.time())
    )
    
    # Statistics
    stats = db.get_attendance_statistics(selected_student.id)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tổng buổi", stats['total'])
    with col2:
        st.metric("Có mặt", stats['present'])
    with col3:
        st.metric("Đi trễ", stats['late'])
    with col4:
        st.metric("Vắng", stats['absent'])
    
    # Progress bar
    st.progress(stats['attendance_rate'] / 100, text=f"Tỷ lệ tham gia: {stats['attendance_rate']:.1f}%")
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Phân bố trạng thái")
        
        status_data = {
            'Trạng thái': ['Có mặt', 'Đi trễ', 'Vắng'],
            'Số lượng': [stats['present'], stats['late'], stats['absent']]
        }
        
        fig = px.bar(
            status_data,
            x='Trạng thái',
            y='Số lượng',
            color='Trạng thái',
            color_discrete_map={
                'Có mặt': '#00CC96',
                'Đi trễ': '#FFA15A',
                'Vắng': '#EF553B'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Lịch sử điểm danh")
        
        if history:
            # Group by date
            attendance_by_date = {}
            for att in history:
                date = att.session.date.date()
                if date not in attendance_by_date:
                    attendance_by_date[date] = []
                attendance_by_date[date].append(att.status)
            
            dates = sorted(attendance_by_date.keys())
            statuses = [attendance_by_date[d][0] for d in dates]  # First status of the day
            
            status_map = {'present': 1, 'late': 0.5, 'absent': 0}
            values = [status_map[s] for s in statuses]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines+markers',
                name='Attendance',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                yaxis=dict(
                    tickmode='array',
                    tickvals=[0, 0.5, 1],
                    ticktext=['Vắng', 'Đi trễ', 'Có mặt']
                ),
                xaxis_title="Ngày",
                yaxis_title="Trạng thái"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Không có dữ liệu trong khoảng thời gian này")
    
    st.divider()
    
    # Detailed history table
    st.subheader("Chi tiết lịch sử")
    
    if history:
        history_data = []
        for att in history:
            history_data.append({
                'Ngày': att.session.date.strftime('%Y-%m-%d'),
                'Môn học': att.session.schedule.subject,
                'Giờ vào': format_time(att.check_in_time),
                'Giờ ra': format_time(att.check_out_time),
                'Trạng thái': att.status.upper(),
                'Ghi chú': att.notes or ''
            })
        
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True, hide_index=True)


def show_schedule(db):
    """Trang quản lý lịch học"""
    st.header("[SCHEDULE] Lịch học")
    
    tab1, tab2 = st.tabs(["Xem lịch", "Thêm mới"])
    
    with tab1:
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        day_names = ['Thứ 2', 'Thứ 3', 'Thứ 4', 'Thứ 5', 'Thứ 6', 'Thứ 7', 'Chủ nhật']
        
        for day, day_name in zip(days, day_names):
            with st.expander(f" {day_name}", expanded=(day == datetime.now().strftime('%A').lower())):
                schedules = db.get_schedule_for_day(day)
                
                if schedules:
                    schedule_data = []
                    for s in schedules:
                        schedule_data.append({
                            'Môn học': s.subject,
                            'Thời gian': f"{s.start_time} - {s.end_time}",
                            'Phòng': s.room,
                            'Giáo viên': s.teacher
                        })
                    
                    df = pd.DataFrame(schedule_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("Không có lịch học")
    
    with tab2:
        with st.form("add_schedule"):
            st.subheader("Thêm lịch học mới")
            
            col1, col2 = st.columns(2)
            
            with col1:
                subject = st.text_input("Môn học *")
                day = st.selectbox("Thứ *", 
                    ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
                    format_func=lambda x: ['Thứ 2', 'Thứ 3', 'Thứ 4', 'Thứ 5', 'Thứ 6', 'Thứ 7', 'Chủ nhật'][
                        ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'].index(x)
                    ]
                )
            
            with col2:
                start_time = st.time_input("Giờ bắt đầu *")
                end_time = st.time_input("Giờ kết thúc *")
            
            room = st.text_input("Phòng học")
            teacher = st.text_input("Giáo viên")
            
            submitted = st.form_submit_button(" Thêm lịch học")
            
            if submitted:
                if not subject:
                    st.error("Vui lòng điền môn học!")
                else:
                    result = db.add_schedule(
                        subject=subject,
                        day_of_week=day,
                        start_time=start_time.strftime('%H:%M'),
                        end_time=end_time.strftime('%H:%M'),
                        room=room if room else None,
                        teacher=teacher if teacher else None
                    )
                    
                    if result:
                        st.success(f"[SUCCESS] Đã thêm lịch học: {subject}")
                        st.rerun()
                    else:
                        st.error("Lỗi khi thêm lịch học")


def main():
    """Main dashboard"""
    
    # Initialize database
    db = get_database()
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=Logo", width=150)
        st.title("Menu")
        
        page = st.radio(
            "Chọn trang",
            [" Tổng quan", " Học sinh", "[LIST] Điểm danh", "[STATS] Báo cáo", "[SCHEDULE] Lịch học"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        st.info(f"**Hôm nay:** {datetime.now().strftime('%d/%m/%Y')}")
        st.info(f"**Thời gian:** {datetime.now().strftime('%H:%M')}")
        
        st.divider()
        
        # Quick actions
        st.subheader("[FAST] Thao tác nhanh")
        
        if st.button("rotate Tải lịch từ config"):
            with st.spinner("Đang tải..."):
                db.load_schedules_from_config()
                st.success("[SUCCESS] Đã tải lịch học!")
                st.rerun()
        
        if st.button("[STATS] Xuất báo cáo"):
            st.info("Tính năng đang phát triển")
    
    # Main content
    if "Tổng quan" in page:
        show_overview(db)
    elif "Học sinh" in page:
        show_students(db)
    elif "Điểm danh" in page:
        show_attendance(db)
    elif "Báo cáo" in page:
        show_reports(db)
    elif "Lịch học" in page:
        show_schedule(db)


if __name__ == "__main__":
    main()
