import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import kagglehub

# --- Configuração da Página ---
st.set_page_config(
    page_title="Análise de Vendas de Videogames",
    page_icon="🎮",
    layout="wide"
)

# --- Título do Dashboard ---
st.title("🎮 Dashboard de Análise de Vendas de Videogames")
st.markdown("""
    Trabalho de Conclusão de Curso do curso de Pós-graduação de Ciência de dados aplicado à inteligência de negócios.
    Exploração de tendências e comparações de vendas de jogos globais.
    Dados de vendas em milhões de unidades.
""")
st.markdown("---")

# --- Mapeamento de Gêneros ---
genre_translation_map = {
    'Action': 'Ação',
    'Sports': 'Esportes',
    'Platform': 'Plataforma',
    'Racing': 'Corrida',
    'Role-Playing': 'RPG',
    'Misc': 'Diversos',
    'Simulation': 'Simulação',
    'Shooter': 'Tiro',
    'Adventure': 'Aventura',
    'Fighting': 'Luta',
    'Strategy': 'Estratégia',
    'Puzzle': 'Quebra-cabeça'
}

# --- Carregamento e Pré-processamento dos Dados ---
@st.cache_data # Armazena os dados em cache para não recarregar a cada interação
def load_data(): 
    try:
        dataset_folder_path = kagglehub.dataset_download("gregorut/videogamesales")
        csv_file_path = os.path.join(dataset_folder_path, 'vgsales.csv')

        df = pd.read_csv(csv_file_path) 
        df.columns = df.columns.str.strip() 
        
        df = df.rename(columns={
            'Name': 'Nome',
            'Platform': 'Console',
            'Year': 'Ano',
            'Genre': 'Genero',
            'Publisher': 'Editora',
            'NA_Sales': 'vendas_na',
            'EU_Sales': 'vendas_eu',
            'JP_Sales': 'vendas_jp',
            'Other_Sales': 'vendas_outros',
            'Global_Sales': 'vendas_globais'
        }, errors='raise') 

        required_cols_after_rename = ['Ano', 'Editora', 'Genero', 'Console', 'vendas_na', 'vendas_eu', 'vendas_jp', 'vendas_outros', 'vendas_globais', 'Nome']
        missing_cols = [col for col in required_cols_after_rename if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Após renomear, as seguintes colunas essenciais ainda estão faltando: {missing_cols}. Colunas atuais no DataFrame: {df.columns.tolist()}")

        df = df.dropna(subset=['Ano', 'Editora']) 
        df['Ano'] = df['Ano'].astype(int) 
        
        numerical_cols = ['vendas_na', 'vendas_eu', 'vendas_jp', 'vendas_outros', 'vendas_globais']
        for col in numerical_cols:
            if col in df.columns: 
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=numerical_cols) 

        df['Genero'] = df['Genero'].replace(genre_translation_map)
        
        return df
    except FileNotFoundError:
        st.error("Erro: O arquivo 'vgsales.csv' não foi encontrado dentro do dataset baixado do Kaggle. Verifique o nome do arquivo.")
        st.stop()
    except KeyError as e:
        st.error(f"Erro de coluna após o download ou renomeação: {e}. Isso pode indicar que o arquivo CSV baixado do Kaggle tem nomes de colunas diferentes do esperado ou está malformado. Colunas fornecidas pelo usuário para referência: Rank, Name, Platform, Year, Genre, Publisher, NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales.")
        return pd.DataFrame()
    except Exception as e:
        if "AuthenticationError" in str(e) or "kaggle.json" in str(e):
            st.error("Erro de autenticação do Kaggle. Por favor, certifique-se de que seu arquivo 'kaggle.json' está configurado corretamente em '~/.kaggle/' (Linux/macOS) ou 'C:\\Users\\<seu_usuario>\\.kaggle\\' (Windows).")
        elif "HTTPError" in str(e) or "DatasetNotFound" in str(e):
            st.error(f"Erro ao baixar o dataset do Kaggle. Verifique o nome do dataset ('gregorut/videogamesales') e sua conexão com a internet. Detalhes: {e}")
        else:
            st.error(f"Ocorreu um erro inesperado ao carregar ou processar os dados: {e}. Isso pode ser um problema com o formato dos dados ou a conexão.")
        return pd.DataFrame()

df = load_data()

# Verifica se o DataFrame foi carregado com sucesso
if df.empty:
    st.stop()

# --- Cálculos pré-computados (executados apenas uma vez) ---
sales_by_year_sum = df.groupby('Ano')['vendas_globais'].sum().reset_index()
top_3_years = sales_by_year_sum.nlargest(3, 'vendas_globais')

# --- Filtros Globais na Sidebar ---
st.sidebar.header("⚙️ Filtros")

min_year = int(df['Ano'].min())
max_year = int(df['Ano'].max())

# Inicializa 'selected_years_peak' no session_state se não existir
if 'selected_years_peak' not in st.session_state:
    st.session_state['selected_years_peak'] = (min_year, max_year)

# Slider para seleção de ano. Ele reflete o estado do 'selected_years_peak'.
selected_years = st.sidebar.slider(
    "Selecione o Intervalo de Anos:",
    min_year, max_year, 
    st.session_state['selected_years_peak']
)
# Se o slider for movido manualmente e for diferente do session_state, atualiza o session_state
if selected_years != st.session_state['selected_years_peak']:
    st.session_state['selected_years_peak'] = selected_years

def create_multiselect_filter(df, column_name, header, session_state_key, emoji=""):
    """Cria um filtro multiselect na sidebar com botões de 'Limpar' e 'Adicionar todos'."""
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {emoji} Filtro de {header}")
    
    all_items = sorted(df[column_name].unique())
    if session_state_key not in st.session_state:
        st.session_state[session_state_key] = all_items

    col1, col2 = st.sidebar.columns(2)
    if col1.button(f"Limpar todos", key=f"clear_{session_state_key}"):
        st.session_state[session_state_key] = []
        st.rerun()
    if col2.button(f"Adicionar todos", key=f"add_all_{session_state_key}"):
        st.session_state[session_state_key] = all_items
        st.rerun()

    selected_items = st.sidebar.multiselect(
        f"Selecione os {header}:",
        all_items,
        default=st.session_state[session_state_key],
        key=f"{session_state_key}_multiselect"
    )

    if selected_items != st.session_state[session_state_key]:
        st.session_state[session_state_key] = selected_items
        st.rerun()
    
    return selected_items

# Usando a função refatorada para criar os filtros
selected_platforms = create_multiselect_filter(df, 'Console', 'Plataformas', 'selected_platforms_state', '🎮')
selected_genres = create_multiselect_filter(df, 'Genero', 'Gêneros', 'selected_genres_state', '🎲')

# O DataFrame filtrado agora usa as variáveis selected_platforms e selected_genres que são controladas pelo session_state
df_filtered = df[
    (df['Ano'].between(selected_years[0], selected_years[1])) &
    (df['Console'].isin(selected_platforms)) &
    (df['Genero'].isin(selected_genres))
]

if df_filtered.empty:
    st.warning("Nenhum dado encontrado com os filtros selecionados. Por favor, ajuste os filtros.")
    st.stop()

# --- Métricas Principais ---
st.subheader("💡 Métricas Principais")
col1, col2, col3, col4 = st.columns(4)

total_sales = df_filtered['vendas_globais'].sum()
avg_sales_per_game = df_filtered['vendas_globais'].mean()
num_games = df_filtered['Nome'].nunique()
num_publishers = df_filtered['Editora'].nunique()

with col1:
    st.metric("Total de Vendas Globais", f"{total_sales:,.2f} M", help="Total de vendas mundiais (em milhões)")
with col2:
    st.metric("Venda Média por Jogo", f"{avg_sales_per_game:,.2f} M", help="Média de vendas globais por jogo")
with col3:
    st.metric("Número de Jogos Únicos", f"{num_games:,}")
with col4:
    st.metric("Número de Editoras", f"{num_publishers:,}")

st.markdown("---")

# --- Gráficos de Comparação ---

# 1. Vendas Globais por Gênero
st.subheader("📊 Vendas Globais por Gênero", help="Compara o desempenho de vendas globais entre diferentes gêneros de jogos. Ajuda a identificar quais gêneros são mais populares em vendas.")
sales_by_genre = df_filtered.groupby('Genero')['vendas_globais'].sum().reset_index()
sales_by_genre = sales_by_genre.sort_values('vendas_globais', ascending=False)

fig_genre = px.bar(
    sales_by_genre,
    x='vendas_globais',
    y='Genero',
    orientation='h',
    title='Vendas Globais Totais por Gênero (Milhões)',
    labels={'vendas_globais': 'Total de Vendas Mundiais (Milhões)', 'Genero': 'Gênero'},
    color='vendas_globais',
    color_continuous_scale=px.colors.sequential.Plasma,
    text='vendas_globais'
)
fig_genre.update_traces(texttemplate='%{text:,.2f}', textposition='outside')
fig_genre.update_layout(
    yaxis={'categoryorder':'total ascending'},
    xaxis_range=[0, sales_by_genre['vendas_globais'].max() * 1.1] # 10% a mais para o texto
)
st.plotly_chart(fig_genre, use_container_width=True)

st.markdown("---")

# 2. Vendas Globais por Plataforma
st.subheader("🎮 Vendas Globais por Plataforma", help="Compara as vendas globais entre as principais plataformas de jogos. Ajuda a entender quais consoles ou sistemas geraram mais receita.")
sales_by_platform = df_filtered.groupby('Console')['vendas_globais'].sum().reset_index()
sales_by_platform = sales_by_platform.sort_values('vendas_globais', ascending=False).head(15)

fig_platform = px.bar(
    sales_by_platform,
    x='vendas_globais',
    y='Console',
    orientation='h',
    title='Top 15 Plataformas por Vendas Globais (Milhões)',
    labels={'vendas_globais': 'Total de Vendas Mundiais (Milhões)', 'Console': 'Plataforma'},
    color='vendas_globais',
    color_continuous_scale=px.colors.sequential.Viridis,
    text='vendas_globais'
)
fig_platform.update_traces(texttemplate='%{text:,.2f}', textposition='outside')
fig_platform.update_layout(
    yaxis={'categoryorder':'total ascending'},
    xaxis_range=[0, sales_by_platform['vendas_globais'].max() * 1.1] # 10% a mais para o texto
)
st.plotly_chart(fig_platform, use_container_width=True)

st.markdown("---")

# 3. Vendas Globais por Ano (Tendência ao Longo do Tempo)
st.subheader("📈 Tendência de Vendas Globais por Ano", help="Mostra a evolução das vendas globais de jogos ao longo dos anos. Ideal para identificar períodos de crescimento ou declínio na indústria.")
sales_by_year_trend = df_filtered.groupby('Ano')['vendas_globais'].sum().reset_index()

fig_year_trend = px.line(
    sales_by_year_trend,
    x='Ano',
    y='vendas_globais',
    title='Vendas Globais Totais por Ano (Milhões)',
    labels={'vendas_globais': 'Total de Vendas Mundiais (Milhões)', 'Ano': 'Ano'},
    markers=True
)
fig_year_trend.update_layout(hovermode="x unified")
st.plotly_chart(fig_year_trend, use_container_width=True)

#Aviso sobre dados incompletos nos anos finais
st.info("""
    **Observação sobre a tendência:** A queda nas vendas nos anos mais recentes (a partir de 2017/2018)
    neste gráfico e nas projeções não representa necessariamente um declínio real do mercado.
    É um comportamento comum em datasets históricos como este, onde a coleta de dados
    para os anos mais recentes pode ser **incompleta ou ter sido descontinuada**.
    Para análises de tendências recentes, seria necessário um dataset mais atualizado.
""")
st.markdown("---")

# 4. Top Publishers
st.subheader("🏢 Top 10 Editoras por Vendas Globais", help="Exibe a participação de mercado das 10 editoras de jogos com maior volume de vendas globais. Permite identificar os principais players.")
#Gráfico de barras para Top Publishers
sales_by_publisher = df_filtered.groupby('Editora')['vendas_globais'].sum().reset_index()
sales_by_publisher = sales_by_publisher.sort_values('vendas_globais', ascending=False).head(10)

fig_publisher = px.bar( # Convertido para gráfico de barras
    sales_by_publisher,
    x='vendas_globais',
    y='Editora',
    orientation='h',
    title='Top 10 Editoras por Vendas Globais (Milhões)',
    labels={'vendas_globais': 'Total de Vendas Mundiais (Milhões)', 'Editora': 'Editora'},
    color='vendas_globais',
    color_continuous_scale=px.colors.sequential.RdBu,
    text='vendas_globais'
)
fig_publisher.update_traces(texttemplate='%{text:,.2f}', textposition='outside')
fig_publisher.update_layout(
    yaxis={'categoryorder':'total ascending'},
    xaxis_range=[0, sales_by_publisher['vendas_globais'].max() * 1.1] # 10% a mais para o texto
)
st.plotly_chart(fig_publisher, use_container_width=True)

st.markdown("---")

# 5. Vendas Regionais (Comparação de Regiões)
st.subheader("🌎 Vendas por Região", help="Compara o desempenho de vendas de jogos em diferentes regiões geográficas. Ajuda a entender onde os jogos são mais populares.")

region_sales_data = df_filtered[['vendas_na', 'vendas_eu', 'vendas_jp', 'vendas_outros']].sum().reset_index()
region_sales_data.columns = ['Região', 'Vendas']

region_display_map = {
    'vendas_na': 'América do Norte (NA)',
    'vendas_eu': 'Europa (EU)',
    'vendas_jp': 'Japão (JP)',
    'vendas_outros': 'Outras Regiões'
}
region_sales_data['Região'] = region_sales_data['Região'].map(region_display_map)


fig_regions = px.bar(
    region_sales_data,
    x='Região',
    y='Vendas',
    title='Vendas Globais por Região (Milhões)',
    labels={'Vendas': 'Vendas (Milhões)', 'Região': 'Região'},
    color='Região',
    color_discrete_map={
        'América do Norte (NA)': 'cornflowerblue', 
        'Europa (EU)': 'mediumseagreen', 
        'Japão (JP)': 'indianred', 
        'Outras Regiões': 'darkgoldenrod'
    },
    text='Vendas'
)
fig_regions.update_traces(texttemplate='%{text:,.2f}', textposition='outside')
fig_regions.update_layout(yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig_regions, use_container_width=True)

st.markdown("---")

#Gráfico de Treemap para Vendas Globais Detalhadas por Região e Categoria
st.subheader("🌳 Vendas Globais Detalhadas por Região e Categoria", help="Visualiza a proporção de vendas globais por região e, dentro de cada região, detalha a contribuição de gêneros ou plataformas. Excelente para entender a hierarquia de vendas.")

treemap_detail_by = st.radio(
    "Detalhar Treemap por:",
    options=['Gênero', 'Plataforma'],
    key='treemap_detail_by_radio'
)

# Preparar dados para o Treemap com 'melt' para uma representação precisa
df_treemap_data = df_filtered.melt(
    id_vars=['Nome', 'Genero', 'Console'], # Colunas para manter
    value_vars=['vendas_na', 'vendas_eu', 'vendas_jp', 'vendas_outros'], # Colunas para "derreter"
    var_name='Regiao',
    value_name='Vendas'
)
# Remover linhas onde as vendas regionais são zero e mapear nomes
df_treemap_data = df_treemap_data[df_treemap_data['Vendas'] > 0]
df_treemap_data['Regiao'] = df_treemap_data['Regiao'].map(region_display_map)


if treemap_detail_by == 'Gênero':
    treemap_path = [px.Constant("Vendas Globais"), 'Regiao', 'Genero']
    treemap_labels_map = {'labels': {'Vendas': 'Vendas (Milhões)', 'Genero': 'Gênero', 'Regiao': 'Região'}}
else: # Plataforma
    treemap_path = [px.Constant("Vendas Globais"), 'Regiao', 'Console']
    treemap_labels_map = {'labels': {'Vendas': 'Vendas (Milhões)', 'Console': 'Plataforma', 'Regiao': 'Região'}}

fig_treemap = px.treemap(
    df_treemap_data,
    path=treemap_path,
    values='Vendas',
    color='Vendas',
    color_continuous_scale='Portland',
    title='Vendas Globais por Região e Categoria (Milhões)',
    **treemap_labels_map
)
fig_treemap.update_layout(margin = dict(t=50, l=25, r=25, b=25)) # Ajusta margens
st.plotly_chart(fig_treemap, use_container_width=True)

st.markdown("---")


# --- NOVOS INSIGHTS E FILTROS PERSONALIZADOS ---

# Insight 1: Análise Detalhada de Vendas por Região com Sub-segmentação
st.subheader("🔍 Análise Detalhada por Região", help="Permite explorar as vendas em uma região específica, detalhando-as por gênero, plataforma ou pelos principais jogos. Ótimo para entender o mercado local.")
col_region_insight1, col_region_insight2 = st.columns(2)

with col_region_insight1:
    selected_region_detail = st.selectbox( 
        "Selecione a Região para Detalhar:",
        options=['vendas_na', 'vendas_eu', 'vendas_jp', 'vendas_outros'],
        format_func=lambda x: region_display_map[x] 
    )

with col_region_insight2:
    detail_by = st.radio(
        "Detalhar por:",
        options=['Gênero', 'Plataforma', 'Jogo']
    )

if detail_by == 'Gênero':
    detail_col = 'Genero'
    title_suffix = 'por Gênero'
elif detail_by == 'Plataforma':
    detail_col = 'Console'
    title_suffix = 'por Plataforma'
else: # detail_by == 'Jogo'
    detail_col = 'Nome'
    title_suffix = 'pelos Top Jogos'

# Lógica de agrupamento para os detalhes regionais
if detail_by == 'Jogo':
    num_top_games_regional = st.slider(
        f"Top N Jogos em {region_display_map[selected_region_detail]}:",
        min_value=5, max_value=30, value=10, step=1, key='top_games_regional_slider'
    )
    sales_by_region_detail = df_filtered.sort_values(selected_region_detail, ascending=False).head(num_top_games_regional)
else:
    sales_by_region_detail = df_filtered.groupby(detail_col)[selected_region_detail].sum().reset_index()
    sales_by_region_detail = sales_by_region_detail.sort_values(selected_region_detail, ascending=False)


fig_region_detail = px.bar(
    sales_by_region_detail,
    x=selected_region_detail,
    y=detail_col,
    orientation='h',
    title=f'Vendas em {region_display_map[selected_region_detail]} {title_suffix} (Milhões)',
    labels={selected_region_detail: f'Vendas em {region_display_map[selected_region_detail]} (Milhões)', detail_col: detail_by},
    color=selected_region_detail,
    color_continuous_scale=px.colors.sequential.YlGnBu,
    text=selected_region_detail
)
fig_region_detail.update_traces(texttemplate='%{text:,.2f}', textposition='outside')
fig_region_detail.update_layout(
    yaxis={'categoryorder':'total ascending'},
    xaxis_range=[0, sales_by_region_detail[selected_region_detail].max() * 1.1] # 10% a mais para o texto
)
st.plotly_chart(fig_region_detail, use_container_width=True)

st.markdown("---")

# Insight 2: Anos de Pico de Vendas
st.subheader("🎯 Anos de Pico de Vendas", help="Identifica e permite focar nos anos com as maiores vendas globais de jogos, revelando períodos de alta na indústria.")
st.markdown("Clique nos botões para focar nos anos de maior venda global.")

cols_top_years = st.columns(3)
for i, (index, row) in enumerate(top_3_years.iterrows()):
    with cols_top_years[i]:
        if st.button(f"Ano: {int(row['Ano'])} ({row['vendas_globais']:.2f} M)", key=f"peak_year_{row['Ano']}"):
            st.session_state['selected_years_peak'] = (int(row['Ano']), int(row['Ano']))
            st.rerun() 

# Se um ano de pico foi selecionado, exibe o gráfico detalhado para aquele ano
if st.session_state['selected_years_peak'] != (min_year, max_year):
    st.info(f"Exibindo dados focados no ano: **{st.session_state['selected_years_peak'][0]}**")
    
    sales_by_genre_peak = df_filtered.groupby('Genero')['vendas_globais'].sum().reset_index()
    sales_by_genre_peak = sales_by_genre_peak.sort_values('vendas_globais', ascending=False)
    
    fig_genre_peak = px.bar(
        sales_by_genre_peak,
        x='vendas_globais',
        y='Genero',
        orientation='h',
        title=f'Vendas por Gênero em {st.session_state["selected_years_peak"][0]} (Milhões)',
        labels={'vendas_globais': 'Total de Vendas Mundiais (Milhões)', 'Genero': 'Gênero'},
        color='vendas_globais',
        color_continuous_scale=px.colors.sequential.Plotly3,
        text='vendas_globais'
    )
    fig_genre_peak.update_traces(texttemplate='%{text:,.2f}', textposition='outside')
    fig_genre_peak.update_layout(
        yaxis={'categoryorder':'total ascending'},
        xaxis_range=[0, sales_by_genre_peak['vendas_globais'].max() * 1.1] # 10% a mais para o texto
    )
    st.plotly_chart(fig_genre_peak, use_container_width=True)

    if st.button("Remover Filtro de Ano de Pico", key="remove_peak_filter"):
        st.session_state['selected_years_peak'] = (min_year, max_year) # Reseta para todos os anos
        st.rerun() 

st.markdown("---")

# Insight 3: Comparativo entre Gêneros/Plataformas Selecionados
st.subheader("🆚 Comparativo Personalizado", help="Permite comparar a evolução das vendas globais de até 3 gêneros ou plataformas selecionados, revelando suas tendências e desempenho relativo ao longo do tempo.")
compare_by_options = ['Genero', 'Console']
compare_by_selected = st.selectbox(
    "Comparar por:",
    options=compare_by_options,
    key='compare_by_selector'
)

# Obter as opções de seleção com base no que será comparado
if compare_by_selected == 'Genero':
    available_items = sorted(df_filtered['Genero'].unique())
else:
    available_items = sorted(df_filtered['Console'].unique())

selected_items_to_compare = st.multiselect(
    f"Selecione até 3 {compare_by_selected}s para comparar:",
    available_items,
    max_selections=3,
    key='items_to_compare_multiselect'
)

if len(selected_items_to_compare) > 0:
    df_compare = df_filtered[df_filtered[compare_by_selected].isin(selected_items_to_compare)]
    
    sales_compare = df_compare.groupby([compare_by_selected, 'Ano'])['vendas_globais'].sum().reset_index()

    fig_compare = px.line(
        sales_compare,
        x='Ano',
        y='vendas_globais',
        color=compare_by_selected,
        title=f'Vendas Globais Anuais para {compare_by_selected}s Selecionados (Milhões)',
        labels={'vendas_globais': 'Total de Vendas Mundiais (Milhões)', 'Ano': 'Ano'},
        markers=True,
        line_shape="linear",
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    fig_compare.update_layout(hovermode="x unified")
    st.plotly_chart(fig_compare, use_container_width=True)
else:
    st.info("Selecione 1 a 3 itens para ver o comparativo.")

st.markdown("---")

# Insight 4: Top Jogos por Vendas Globais (TOP 20 FIXO)
st.subheader("🏆 Top Jogos por Vendas Globais", help="Exibe os 20 jogos individuais com as maiores vendas globais dentro dos filtros aplicados. Ideal para identificar os maiores blockbusters.")

top_n_games = 20 
top_games_data = df_filtered.sort_values('vendas_globais', ascending=False).head(top_n_games)

if not top_games_data.empty: # Garante que há dados para plotar
    fig_top_games = px.bar(
        top_games_data,
        x='vendas_globais',
        y='Nome',
        orientation='h',
        title=f'Top {top_n_games} Jogos por Vendas Globais (Milhões)',
        labels={'vendas_globais': 'Total de Vendas Mundiais (Milhões)', 'Nome': 'Nome do Jogo'},
        color='vendas_globais', # Usando a coluna de vendas para cor contínua
        color_continuous_scale=px.colors.sequential.Plotly3, # Exemplo de escala de cor
        text='vendas_globais'
    )
    fig_top_games.update_traces(texttemplate='%{text:,.2f}', textposition='outside')
    fig_top_games.update_layout(
        yaxis={'categoryorder':'total ascending'},
        xaxis_range=[0, top_games_data['vendas_globais'].max() * 1.1], # 10% a mais para o texto
        coloraxis_showscale=True # Mantém a barrinha de cores visível (para Top Jogos)
    )
    st.plotly_chart(fig_top_games, use_container_width=True)
else:
    st.info(f"Nenhum jogo encontrado no Top {top_n_games} com os filtros atuais.")


st.markdown("---")

# NOVIDADE: Insight de Participação de Mercado por Categoria ao Longo do Tempo (Substitui Projeções)
st.subheader("📊 Participação de Mercado por Categoria ao Longo do Tempo", help="Mostra como a contribuição percentual de cada Gênero ou Plataforma para as vendas globais totais evoluiu ao longo dos anos. Ajuda a identificar tendências de dominância de mercado.")

market_share_category = st.radio(
    "Analisar Participação de Mercado por:",
    options=['Gênero', 'Plataforma'],
    key='market_share_category_radio'
)

if market_share_category == 'Gênero':
    market_share_col_name = 'Genero'
else:
    market_share_col_name = 'Console'

# Calcular vendas totais por ano
total_sales_per_year = df_filtered.groupby('Ano')['vendas_globais'].sum().reset_index()
total_sales_per_year.rename(columns={'vendas_globais': 'Total_Ano'}, inplace=True)

# Calcular vendas por categoria por ano
sales_by_category_year = df_filtered.groupby([market_share_col_name, 'Ano'])['vendas_globais'].sum().reset_index()

# Juntar para calcular a participação de mercado
df_market_share = pd.merge(sales_by_category_year, total_sales_per_year, on='Ano')
df_market_share['Participacao_Mercado'] = (df_market_share['vendas_globais'] / df_market_share['Total_Ano']) * 100

# Filtro para as categorias a serem exibidas na participação de mercado
available_ms_categories = sorted(df_market_share[market_share_col_name].unique())
# Padrão: top N por vendas globais totais para o filtro de participação de mercado
if not df_filtered.empty:
    top_n_for_ms_default = df_filtered.groupby(market_share_col_name)['vendas_globais'].sum().nlargest(min(7, len(available_ms_categories))).index.tolist()
else:
    top_n_for_ms_default = []

selected_ms_categories = st.multiselect(
    f"Selecione {market_share_category}s para visualizar a Participação de Mercado:",
    options=available_ms_categories,
    default=top_n_for_ms_default if top_n_for_ms_default else available_ms_categories,
    key='select_ms_categories_multiselect'
)

if selected_ms_categories:
    df_market_share_filtered = df_market_share[df_market_share[market_share_col_name].isin(selected_ms_categories)]
    
    fig_market_share = px.area(
        df_market_share_filtered,
        x='Ano',
        y='Participacao_Mercado',
        color=market_share_col_name,
        title=f'Participação de Mercado por {market_share_category} ao Longo do Tempo (%)',
        labels={'Participacao_Mercado': 'Participação de Mercado (%)', 'Ano': 'Ano'},
        groupnorm='percent', # Garante que as áreas empilhadas somem 100%
        hover_data={'Participacao_Mercado': ':.2f%'}
    )
    fig_market_share.update_layout(hovermode="x unified", yaxis_range=[0, 100]) # Garante que Y vá de 0 a 100
    st.plotly_chart(fig_market_share, use_container_width=True)
else:
    st.info("Selecione pelo menos um item para visualizar a participação de mercado.")

st.markdown("---")

# Insight 5: Análise de Editora Específica
st.subheader("🏢 Análise de Editora Específica", help="Permite selecionar uma editora para ver o detalhamento de suas vendas por gênero e plataforma, ideal para analisar o portfólio de um player específico.")

all_publishers = sorted(df['Editora'].unique())
selected_publisher_for_detail = st.selectbox(
    "Selecione uma Editora para Análise Detalhada:",
    options=['Selecione uma Editora'] + list(all_publishers),
    key='publisher_detail_selector'
)

if selected_publisher_for_detail != 'Selecione uma Editora':
    df_publisher_detail = df_filtered[df_filtered['Editora'] == selected_publisher_for_detail]
    
    if not df_publisher_detail.empty:
        st.markdown(f"**Vendas Totais da {selected_publisher_for_detail}:** {df_publisher_detail['vendas_globais'].sum():,.2f} M")
        
        # Detalhe por Gênero para a Editora
        sales_by_publisher_genre = df_publisher_detail.groupby('Genero')['vendas_globais'].sum().reset_index()
        sales_by_publisher_genre = sales_by_publisher_genre.sort_values('vendas_globais', ascending=False)

        fig_pub_genre = px.bar(
            sales_by_publisher_genre,
            x='vendas_globais',
            y='Genero',
            orientation='h',
            title=f'Vendas por Gênero para {selected_publisher_for_detail} (Milhões)',
            labels={'vendas_globais': 'Total de Vendas Mundiais (Milhões)', 'Genero': 'Gênero'},
            color='vendas_globais',
            color_continuous_scale=px.colors.sequential.OrRd,
            text='vendas_globais'
        )
        fig_pub_genre.update_traces(texttemplate='%{text:,.2f}', textposition='outside')
        fig_pub_genre.update_layout(
            yaxis={'categoryorder':'total ascending'},
            xaxis_range=[0, sales_by_publisher_genre['vendas_globais'].max() * 1.1] # 10% a mais para o texto
        )
        st.plotly_chart(fig_pub_genre, use_container_width=True)

        # Detalhe por Plataforma para a Editora
        sales_by_publisher_platform = df_publisher_detail.groupby('Console')['vendas_globais'].sum().reset_index()
        sales_by_publisher_platform = sales_by_publisher_platform.sort_values('vendas_globais', ascending=False).head(10)

        fig_pub_platform = px.bar(
            sales_by_publisher_platform,
            x='vendas_globais',
            y='Console',
            orientation='h',
            title=f'Vendas por Plataforma para {selected_publisher_for_detail} (Milhões)',
            labels={'vendas_globais': 'Total de Vendas Mundiais (Milhões)', 'Console': 'Plataforma'},
            color='vendas_globais',
            color_continuous_scale=px.colors.sequential.Blues,
            text='vendas_globais'
        )
        fig_pub_platform.update_traces(texttemplate='%{text:,.2f}', textposition='outside')
        fig_pub_platform.update_layout(
            yaxis={'categoryorder':'total ascending'},
            xaxis_range=[0, sales_by_publisher_platform['vendas_globais'].max() * 1.1] # 10% a mais para o texto
        )
        st.plotly_chart(fig_pub_platform, use_container_width=True)

    else:
        st.info("Nenhum dado para a editora selecionada com os filtros atuais.")

st.markdown("---")

# --- Tabela Detalhada (Expansível) ---
st.subheader("📋 Dados Detalhados dos Jogos", help="Exibe a tabela completa dos dados, aplicando todos os filtros selecionados, permitindo uma inspeção detalhada dos registros.")
with st.expander("Clique para ver a Tabela Completa"):
    st.dataframe(df_filtered, use_container_width=True)

st.markdown("---")

# --- Seção "Sobre" o Dashboard ---
st.subheader("Sobre este Dashboard")
st.info(f"""
Este dashboard foi desenvolvido por **Wilber Soares** para analisar dados de vendas de videogames como trabalho de conclusão de curso do curso de Pós-graduação de Ciência de dados aplicado à inteligência de negócios.
Ele permite explorar tendências, comparar performance de gêneros, plataformas e editoras ao longo dos anos.
Dados de vendas são apresentados em milhões de unidades e foram pré-processados para garantir a qualidade da análise.
""")

st.markdown("*Dashboard de Vendas de Videogames | Desenvolvido com Streamlit por Wilber Soares* 🚀")