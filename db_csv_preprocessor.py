import os
import argparse
import pandas as pd
from pathlib import Path

def process_db_dumps(sensor_data_path, sensor_info_path, facility_info_path, output_path):
    print("=" * 60)
    print(" 대용량 DB CSV 전처리 파이프라인 (213만 건 대응)")
    print("=" * 60)
    
    print(f"\n[1] 데이터 로드 중...")
    print(f"  - 시설 정보: {facility_info_path}")
    facility_df = pd.read_csv(facility_info_path)
    
    print(f"  - 센서 메타: {sensor_info_path}")
    sensor_info_df = pd.read_csv(sensor_info_path)
    
    print(f"  - 센서 시계열(대용량): {sensor_data_path}")
    # 메모리 최적화를 위해 필요한 컬럼만, 데이터 타입 지정하여 로드
    sensor_data_df = pd.read_csv(
        sensor_data_path, 
        usecols=['SENSOR_ID', 'WRITE_DATE', 'VALUE'],
        dtype={'SENSOR_ID': 'int32', 'VALUE': 'float32'}
    )
    print(f"    * 로드된 센서 데이터: {len(sensor_data_df):,}건")
    
    print("\n[2] 센서 메타데이터 병합 중 (JOIN)...")
    merged_df = sensor_data_df.merge(
        sensor_info_df[['SENSOR_ID', 'TRANSMITTER_ID', 'NAME']], 
        on='SENSOR_ID', 
        how='left'
    )
    # 더 이상 필요 없는 원본 데이터 메모리 해제
    del sensor_data_df
    
    print("\n[3] 시계열 데이터 피벗(Pivot) 변환 중...")
    print("  - Long Format -> Wide Format (인덱스: 장비, 측정시각 / 컬럼: 센서 종류)")
    
    # 중복 데이터 방지를 위해 mean 집계 사용
    pivot_df = merged_df.pivot_table(
        index=['TRANSMITTER_ID', 'WRITE_DATE'], 
        columns='NAME', 
        values='VALUE',
        aggfunc='mean'
    ).reset_index()
    del merged_df
    
    print("\n[4] 컬럼명 정제 및 시설 정보 매핑 중...")
    # AI 모델이 기대하는 피처명으로 변경
    rename_mapping = {
        '수신감도': '통신품질',
        '방식전류': '희생전류',
        'WRITE_DATE': '측정시각'
    }
    pivot_df = pivot_df.rename(columns=rename_mapping)
    pivot_df['측정시각'] = pd.to_datetime(pivot_df['측정시각'], errors='coerce')
    
    # 장비번호 생성 (예: TR_6) -> 향후 config.json의 sacrificial_devices와 일치시켜야 함
    pivot_df['장비번호'] = 'TR_' + pivot_df['TRANSMITTER_ID'].astype(str)
    
    facility_df = facility_df.rename(columns={
        'NUMBER': '시설번호',
        'POSITION': '주소',
        'LATITUDE': '위도',
        'LONGITUDE': '경도'
    })
    facility_df['장비번호'] = 'TR_' + facility_df['TRANSMITTER_ID'].astype(str)
    
    # 시설 정보 조인
    final_df = pivot_df.merge(
        facility_df[['장비번호', '시설번호', '주소', '위도', '경도']], 
        on='장비번호', 
        how='left'
    )
    
    final_df = final_df.drop(columns=['TRANSMITTER_ID'], errors='ignore')
    final_df = final_df.sort_values(['장비번호', '측정시각']).reset_index(drop=True)
    
    print(f"\n[5] 최종 데이터셋 저장 중... (총 {len(final_df):,}행)")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.parquet':
        final_df.to_parquet(output_path, index=False)
    else:
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
    print(f"  -> 저장 완료: {output_path}")
    print("\n※ [중요] AI 모델 학습 시 다음 사항을 확인하세요:")
    print("  1. 모델의 config.json (또는 스크립트 실행 인자)에서 `excel_filename`을 이 파일 경로로 변경하세요.")
    print("  2. `sacrificial_devices` 설정을 엑셀 장비번호(TB24-...)에서 DB 장비번호(TR_...)로 맞추셔야 합니다.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DB CSV 덤프 병합 및 AI 모델용 전처리')
    parser.add_argument('--data', required=True, help='kscg_sensor_data CSV 파일 경로')
    parser.add_argument('--sensor', required=True, help='kscg_sensor_info CSV 파일 경로')
    parser.add_argument('--facility', required=True, help='kscg_facility_info CSV 파일 경로')
    parser.add_argument('--out', default='preprocessed_master_data.parquet', help='결과 저장 경로 (Parquet 권장)')
    args = parser.parse_args()
    
    process_db_dumps(args.data, args.sensor, args.facility, args.out)