// SpaceProx.cpp
// Small standalone D3D11 top-down arena game used by the black-box RL experiment.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi.h>

#include "Player.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

namespace SpaceProx
{
constexpr wchar_t kWindowClass[] = L"SpaceProxWindowClass";
constexpr wchar_t kWindowTitle[] = L"SpaceProx - D3D11 Arena";
constexpr float kWidth = 960.0f;
constexpr float kHeight = 640.0f;
constexpr float kPi = 3.1415926535f;
constexpr size_t kMaximumVertices = 32768;

struct Color
{
    float r, g, b, a;
};

struct Vertex
{
    float x, y, z;
    float r, g, b, a;
};

struct Rect
{
    float left, top, right, bottom;
};

struct Enemy
{
    float x, y;
    float velocityX, velocityY;
    float radius;
    float hitFlash;
    float respawnTime;
    int value;
    bool alive;
};

struct Bullet
{
    float x, y;
    float velocityX, velocityY;
    float life;
    bool alive;
};

constexpr std::array<Rect, 3> kWalls =
{
    Rect{ 330.0f, 150.0f, 630.0f, 190.0f },
    Rect{ 170.0f, 350.0f, 410.0f, 390.0f },
    Rect{ 650.0f, 385.0f, 790.0f, 425.0f },
};

std::array<Enemy, 7> gEnemies{};
std::array<Bullet, 48> gBullets{};

bool gIsWasted = false;
bool gIsPaused = false;
int gBestScore = 0;

Player::Player() : health(100), score(0), ammo(24), x(480.0f), y(520.0f), aimX(0.0f), aimY(-1.0f),
    hitFlash(0.0f), fireCooldown(0.0f), reloadTime(0.0f) {}
Player* gPlayer = CreatePlayer();

bool PointInRect(float x, float y, const Rect& r)
{
    return x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;
}

bool SegmentIntersectsRect(float x0, float y0, float x1, float y1, const Rect& r)
{
    // Liang-Barsky line clipping for the game's line-of-sight check.
    const float dx = x1 - x0;
    const float dy = y1 - y0;
    const float p[4] = { -dx, dx, -dy, dy };
    const float q[4] = { x0 - r.left, r.right - x0, y0 - r.top, r.bottom - y0 };
    float enter = 0.0f;
    float exit = 1.0f;

    for (int i = 0; i < 4; ++i)
    {
        if (std::fabs(p[i]) < 0.0001f)
        {
            if (q[i] < 0.0f) return false;
            continue;
        }
        const float t = q[i] / p[i];
        if (p[i] < 0.0f)
        {
            if (t > exit) return false;
            enter = std::max(enter, t);
        }
        else
        {
            if (t < enter) return false;
            exit = std::min(exit, t);
        }
    }
    return enter <= exit;
}

bool IsHiddenByCover(int enemyIndex)
{
    if (enemyIndex < 0 || enemyIndex >= static_cast<int>(gEnemies.size())) return true;
    const Enemy& enemy = gEnemies[enemyIndex];
    for (const Rect& wall : kWalls)
    {
        if (SegmentIntersectsRect(gPlayer->x, gPlayer->y, enemy.x, enemy.y, wall))
            return true;
    }
    return false;
}

void Player::Update(float deltaSeconds)
{
    float directionX = 0.0f;
    float directionY = 0.0f;
    if (GetAsyncKeyState('A') & 0x8000 || GetAsyncKeyState(VK_LEFT) & 0x8000) directionX -= 1.0f;
    if (GetAsyncKeyState('D') & 0x8000 || GetAsyncKeyState(VK_RIGHT) & 0x8000) directionX += 1.0f;
    if (GetAsyncKeyState('W') & 0x8000 || GetAsyncKeyState(VK_UP) & 0x8000) directionY -= 1.0f;
    if (GetAsyncKeyState('S') & 0x8000 || GetAsyncKeyState(VK_DOWN) & 0x8000) directionY += 1.0f;

    const float length = std::sqrt(directionX * directionX + directionY * directionY);
    if (length > 0.0f)
    {
        directionX /= length;
        directionY /= length;
        const float speed = 240.0f * GetSpeed(); // virtual: speed hook is visible in-game
        x += directionX * speed * deltaSeconds;
        y += directionY * speed * deltaSeconds;
    }

    x = std::clamp(x, 24.0f, kWidth - 24.0f);
    y = std::clamp(y, 70.0f, kHeight - 24.0f);
    for (const Rect& wall : kWalls)
    {
        if (PointInRect(x, y, wall))
        {
            const float leftDistance = std::fabs(x - wall.left);
            const float rightDistance = std::fabs(wall.right - x);
            const float topDistance = std::fabs(y - wall.top);
            const float bottomDistance = std::fabs(wall.bottom - y);
            const float nearest = std::min({ leftDistance, rightDistance, topDistance, bottomDistance });
            if (nearest == leftDistance) x = wall.left - 1.0f;
            else if (nearest == rightDistance) x = wall.right + 1.0f;
            else if (nearest == topDistance) y = wall.top - 1.0f;
            else y = wall.bottom + 1.0f;
        }
    }

    hitFlash = std::max(0.0f, hitFlash - deltaSeconds * 3.0f);
    fireCooldown = std::max(0.0f, fireCooldown - deltaSeconds);
    if (reloadTime > 0.0f)
    {
        reloadTime -= deltaSeconds;
        if (reloadTime <= 0.0f) ammo = 24;
    }
}

void Player::TakeDamage(int amount)
{
    if (health <= 0) return;
    int actualDamage = GetDamageTaken(amount);
    health = std::max(0, health - actualDamage);
    hitFlash = 1.0f;
    if (health <= 0)
    {
        gBestScore = std::max(gBestScore, score);
        gIsPaused = false;
        gIsWasted = true;
    }
}

int Player::GetHealth() { return health; }
bool Player::IsInvincible() { return false; }
float Player::GetSpeed() { return 1.0f; }
bool Player::IsEnemyVisible(int enemyIndex) { return !IsHiddenByCover(enemyIndex); }
int Player::GetScore() { return score; }
int Player::GetAmmo() { return ammo; }

bool Player::ConsumeAmmo()
{
    if (ammo <= 0 || reloadTime > 0.0f) return false;
    --ammo;
    if (ammo == 0) reloadTime = 1.3f;
    return true;
}

int Player::GetDamageTaken(int amount)
{
    int result = amount;
    if (result < 0) result = 0;       // clamp negatives
    if (result > 100) result = 100;   // cap at 100
    return result;
}

// D3D11 resources.
ID3D11Device* gDevice = nullptr;
ID3D11DeviceContext* gContext = nullptr;
IDXGISwapChain* gSwapChain = nullptr;
ID3D11RenderTargetView* gRenderTarget = nullptr;
ID3D11VertexShader* gVertexShader = nullptr;
ID3D11PixelShader* gPixelShader = nullptr;
ID3D11InputLayout* gInputLayout = nullptr;
ID3D11Buffer* gVertexBuffer = nullptr;
ID3D11RasterizerState* gRasterizer = nullptr;
ID3D11BlendState* gBlendState = nullptr;
std::vector<Vertex> gVertices;

const char kVertexShaderSource[] = R"(
struct VSInput { float3 position : POSITION; float4 color : COLOR; };
struct VSOutput { float4 position : SV_POSITION; float4 color : COLOR; };
VSOutput main(VSInput input) {
    VSOutput output;
    output.position = float4(input.position, 1.0f);
    output.color = input.color;
    return output;
})";

const char kPixelShaderSource[] = R"(
struct PSInput { float4 position : SV_POSITION; float4 color : COLOR; };
float4 main(PSInput input) : SV_TARGET { return input.color; })";

void ReleaseD3D()
{
    if (gBlendState) gBlendState->Release();
    if (gRasterizer) gRasterizer->Release();
    if (gVertexBuffer) gVertexBuffer->Release();
    if (gInputLayout) gInputLayout->Release();
    if (gPixelShader) gPixelShader->Release();
    if (gVertexShader) gVertexShader->Release();
    if (gRenderTarget) gRenderTarget->Release();
    if (gSwapChain) gSwapChain->Release();
    if (gContext) gContext->Release();
    if (gDevice) gDevice->Release();
    gBlendState = nullptr; gRasterizer = nullptr; gVertexBuffer = nullptr;
    gInputLayout = nullptr; gPixelShader = nullptr; gVertexShader = nullptr;
    gRenderTarget = nullptr; gSwapChain = nullptr; gContext = nullptr; gDevice = nullptr;
}

bool CreateD3D(HWND window)
{
    DXGI_SWAP_CHAIN_DESC description{};
    description.BufferDesc.Width = static_cast<UINT>(kWidth);
    description.BufferDesc.Height = static_cast<UINT>(kHeight);
    description.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    description.SampleDesc.Count = 1;
    description.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    description.BufferCount = 1;
    description.OutputWindow = window;
    description.Windowed = TRUE;
    description.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    const D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0 };
    D3D_FEATURE_LEVEL selectedLevel{};
    if (FAILED(D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
        featureLevels, ARRAYSIZE(featureLevels), D3D11_SDK_VERSION, &description,
        &gSwapChain, &gDevice, &selectedLevel, &gContext))) return false;

    ID3D11Texture2D* backBuffer = nullptr;
    const HRESULT targetResult = gSwapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer));
    if (FAILED(targetResult) || FAILED(gDevice->CreateRenderTargetView(backBuffer, nullptr, &gRenderTarget)))
    {
        if (backBuffer) backBuffer->Release();
        return false;
    }
    backBuffer->Release();

    ID3DBlob* vertexBlob = nullptr;
    ID3DBlob* pixelBlob = nullptr;
    ID3DBlob* errorBlob = nullptr;
    const HRESULT vertexResult = D3DCompile(kVertexShaderSource, std::strlen(kVertexShaderSource), nullptr, nullptr, nullptr,
        "main", "vs_4_0", 0, 0, &vertexBlob, &errorBlob);
    if (FAILED(vertexResult))
    {
        if (errorBlob) { MessageBoxA(window, static_cast<const char*>(errorBlob->GetBufferPointer()), "SpaceProx vertex shader", MB_ICONERROR); errorBlob->Release(); }
        return false;
    }
    if (errorBlob) { errorBlob->Release(); errorBlob = nullptr; }
    const HRESULT pixelResult = D3DCompile(kPixelShaderSource, std::strlen(kPixelShaderSource), nullptr, nullptr, nullptr,
        "main", "ps_4_0", 0, 0, &pixelBlob, &errorBlob);
    if (FAILED(pixelResult))
    {
        if (errorBlob) { MessageBoxA(window, static_cast<const char*>(errorBlob->GetBufferPointer()), "SpaceProx pixel shader", MB_ICONERROR); errorBlob->Release(); }
        vertexBlob->Release();
        return false;
    }

    const bool shadersCreated = SUCCEEDED(gDevice->CreateVertexShader(vertexBlob->GetBufferPointer(), vertexBlob->GetBufferSize(), nullptr, &gVertexShader)) &&
        SUCCEEDED(gDevice->CreatePixelShader(pixelBlob->GetBufferPointer(), pixelBlob->GetBufferSize(), nullptr, &gPixelShader));
    const D3D11_INPUT_ELEMENT_DESC layout[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    const bool layoutCreated = SUCCEEDED(gDevice->CreateInputLayout(layout, ARRAYSIZE(layout),
        vertexBlob->GetBufferPointer(), vertexBlob->GetBufferSize(), &gInputLayout));
    vertexBlob->Release();
    pixelBlob->Release();
    if (!shadersCreated || !layoutCreated) return false;

    D3D11_BUFFER_DESC bufferDescription{};
    bufferDescription.ByteWidth = static_cast<UINT>(sizeof(Vertex) * kMaximumVertices);
    bufferDescription.Usage = D3D11_USAGE_DYNAMIC;
    bufferDescription.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bufferDescription.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(gDevice->CreateBuffer(&bufferDescription, nullptr, &gVertexBuffer))) return false;

    D3D11_RASTERIZER_DESC rasterizer{};
    rasterizer.FillMode = D3D11_FILL_SOLID;
    rasterizer.CullMode = D3D11_CULL_NONE;
    rasterizer.DepthClipEnable = TRUE;
    if (FAILED(gDevice->CreateRasterizerState(&rasterizer, &gRasterizer))) return false;

    D3D11_VIEWPORT viewport{};
    viewport.TopLeftX = 0.0f;
    viewport.TopLeftY = 0.0f;
    viewport.Width = kWidth;
    viewport.Height = kHeight;
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    gContext->RSSetViewports(1, &viewport);

    D3D11_BLEND_DESC blend{};
    blend.RenderTarget[0].BlendEnable = TRUE;
    blend.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    blend.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    blend.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    blend.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    blend.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
    blend.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    blend.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    return SUCCEEDED(gDevice->CreateBlendState(&blend, &gBlendState));
}

Vertex MakeVertex(float x, float y, Color color)
{
    return { (x / (kWidth * 0.5f)) - 1.0f, 1.0f - (y / (kHeight * 0.5f)), 0.0f, color.r, color.g, color.b, color.a };
}

void AddTriangle(float x0, float y0, float x1, float y1, float x2, float y2, Color color)
{
    if (gVertices.size() + 3 > kMaximumVertices) return;
    gVertices.push_back(MakeVertex(x0, y0, color));
    gVertices.push_back(MakeVertex(x1, y1, color));
    gVertices.push_back(MakeVertex(x2, y2, color));
}

void AddRect(float left, float top, float right, float bottom, Color color)
{
    AddTriangle(left, top, right, top, right, bottom, color);
    AddTriangle(left, top, right, bottom, left, bottom, color);
}

void AddCircle(float x, float y, float radius, Color color, int segments = 18)
{
    for (int i = 0; i < segments; ++i)
    {
        const float a = (static_cast<float>(i) / segments) * 2.0f * kPi;
        const float b = (static_cast<float>(i + 1) / segments) * 2.0f * kPi;
        AddTriangle(x, y, x + std::cos(a) * radius, y + std::sin(a) * radius,
            x + std::cos(b) * radius, y + std::sin(b) * radius, color);
    }
}

void AddLine(float x0, float y0, float x1, float y1, float thickness, Color color)
{
    const float dx = x1 - x0;
    const float dy = y1 - y0;
    const float length = std::sqrt(dx * dx + dy * dy);
    if (length <= 0.01f) return;
    const float offsetX = -dy / length * thickness * 0.5f;
    const float offsetY = dx / length * thickness * 0.5f;
    AddTriangle(x0 + offsetX, y0 + offsetY, x1 + offsetX, y1 + offsetY, x1 - offsetX, y1 - offsetY, color);
    AddTriangle(x0 + offsetX, y0 + offsetY, x1 - offsetX, y1 - offsetY, x0 - offsetX, y0 - offsetY, color);
}

void DrawWorld()
{
    for (float x = 20.0f; x < kWidth; x += 40.0f) AddLine(x, 58.0f, x, kHeight - 16.0f, 1.0f, { 0.08f, 0.14f, 0.22f, 0.45f });
    for (float y = 80.0f; y < kHeight; y += 40.0f) AddLine(16.0f, y, kWidth - 16.0f, y, 1.0f, { 0.08f, 0.14f, 0.22f, 0.45f });
    AddRect(16.0f, 58.0f, kWidth - 16.0f, kHeight - 16.0f, { 0.02f, 0.08f, 0.13f, 0.12f });
    AddLine(16.0f, 58.0f, kWidth - 16.0f, 58.0f, 2.0f, { 0.15f, 0.65f, 0.95f, 0.80f });
    AddLine(16.0f, 58.0f, 16.0f, kHeight - 16.0f, 2.0f, { 0.15f, 0.65f, 0.95f, 0.80f });
    AddLine(kWidth - 16.0f, 58.0f, kWidth - 16.0f, kHeight - 16.0f, 2.0f, { 0.15f, 0.65f, 0.95f, 0.80f });
    AddLine(16.0f, kHeight - 16.0f, kWidth - 16.0f, kHeight - 16.0f, 2.0f, { 0.15f, 0.65f, 0.95f, 0.80f });

    for (const Rect& wall : kWalls)
    {
        AddRect(wall.left + 5.0f, wall.top + 6.0f, wall.right + 5.0f, wall.bottom + 6.0f, { 0.0f, 0.0f, 0.0f, 0.32f });
        AddRect(wall.left, wall.top, wall.right, wall.bottom, { 0.18f, 0.24f, 0.33f, 1.0f });
        AddLine(wall.left, wall.top, wall.right, wall.top, 2.0f, { 0.36f, 0.75f, 0.95f, 0.80f });
        AddLine(wall.left, wall.bottom, wall.right, wall.bottom, 2.0f, { 0.05f, 0.11f, 0.18f, 1.0f });
    }
}
// 5x7 bitmap font for the WASTED overlay. Each glyph is 7 rows of 5 columns.
// '#' = pixel on, '.' = off.  Only the characters we use are defined.
const char* GetGlyph(char c)
{
    switch (c)
    {
    case 'W': return
        "#...#\n"
        "#...#\n"
        "#...#\n"
        "#...#\n"
        "#.#.#\n"
        "##.##\n"
        "#...#";
    case 'A': return
        "..#..\n"
        ".#.#.\n"
        "#...#\n"
        "#####\n"
        "#...#\n"
        "#...#\n"
        "#...#";
    case 'S': return
        ".####\n"
        "#....\n"
        "#....\n"
        ".###.\n"
        "....#\n"
        "....#\n"
        "####.";
    case 'T': return
        "#####\n"
        "..#..\n"
        "..#..\n"
        "..#..\n"
        "..#..\n"
        "..#..\n"
        "..#..";
    case 'E': return
        "#####\n"
        "#....\n"
        "#....\n"
        "####.\n"
        "#....\n"
        "#....\n"
        "#####";
    case 'D': return
        "####.\n"
        "#...#\n"
        "#...#\n"
        "#...#\n"
        "#...#\n"
        "#...#\n"
        "####.";
    case 'P': return
        "####.\n"
        "#...#\n"
        "#...#\n"
        "####.\n"
        "#....\n"
        "#....\n"
        "#....";
    case 'R': return
        "####.\n"
        "#...#\n"
        "#...#\n"
        "####.\n"
        "#.#..\n"
        "#..#.\n"
        "#...#";
    case 'B': return
        "####.\n#...#\n#...#\n####.\n#...#\n#...#\n####.";
    case 'C': return
        ".####\n#....\n#....\n#....\n#....\n#....\n.####";
    case 'G': return
        ".####\n#....\n#....\n#.###\n#...#\n#...#\n.###.";
    case 'M': return
        "#...#\n##.##\n#.#.#\n#.#.#\n#...#\n#...#\n#...#";
    case 'O': return
        ".###.\n#...#\n#...#\n#...#\n#...#\n#...#\n.###.";
    case 'U': return
        "#...#\n#...#\n#...#\n#...#\n#...#\n#...#\n.###.";
    case 'V': return
        "#...#\n#...#\n#...#\n#...#\n#...#\n.#.#.\n..#..";
    case 'Y': return
        "#...#\n#...#\n.#.#.\n..#..\n..#..\n..#..\n..#..";
    case '0': return
        ".###.\n#...#\n#..##\n#.#.#\n##..#\n#...#\n.###.";
    case '1': return
        "..#..\n.##..\n..#..\n..#..\n..#..\n..#..\n.###.";
    case '2': return
        ".###.\n#...#\n....#\n...#.\n..#..\n.#...\n#####";
    case '3': return
        "####.\n....#\n....#\n.###.\n....#\n....#\n####.";
    case '4': return
        "...#.\n..##.\n.#.#.\n#..#.\n#####\n...#.\n...#.";
    case '5': return
        "#####\n#....\n#....\n####.\n....#\n....#\n####.";
    case '6': return
        ".###.\n#....\n#....\n####.\n#...#\n#...#\n.###.";
    case '7': return
        "#####\n....#\n...#.\n..#..\n.#...\n.#...\n.#...";
    case '8': return
        ".###.\n#...#\n#...#\n.###.\n#...#\n#...#\n.###.";
    case '9': return
        ".###.\n#...#\n#...#\n.####\n....#\n....#\n.###.";
    case ' ': return
        ".....\n"
        ".....\n"
        ".....\n"
        ".....\n"
        ".....\n"
        ".....\n"
        ".....";
    default: return
        ".....\n"
        ".....\n"
        ".....\n"
        ".....\n"
        ".....\n"
        ".....\n"
        ".....";
    }
}

void DrawText(const char* text, float x, float y, float pixel, Color color)
{
    for (const char* cursor = text; *cursor != '\0'; ++cursor)
    {
        const char* glyph = GetGlyph(*cursor);
        for (int row = 0; row < 7; ++row)
        {
            for (int col = 0; col < 5; ++col)
            {
                if (glyph[row * 6 + col] == '#')
                {
                    AddRect(x + col * pixel, y + row * pixel,
                        x + (col + 1) * pixel, y + (row + 1) * pixel, color);
                }
            }
        }
        x += 6 * pixel;
    }
}

void DrawHud()
{
    const int health = std::clamp(gPlayer->GetHealth(), 0, 100); // virtual
    const int ammo = std::clamp(gPlayer->GetAmmo(), 0, 24);      // virtual
    const bool overrideActive = gPlayer->IsInvincible();          // virtual
    const float speed = gPlayer->GetSpeed();                       // virtual

    AddRect(16.0f, 15.0f, 255.0f, 43.0f, { 0.03f, 0.08f, 0.14f, 0.95f });

    AddRect(20.0f, 20.0f, 220.0f, 31.0f, { 0.11f, 0.15f, 0.20f, 1.0f });
    const Color healthColor = health > 50 ? Color{ 0.10f, 0.95f, 0.37f, 1.0f } : Color{ 1.0f, 0.18f, 0.16f, 1.0f };
    AddRect(20.0f, 20.0f, 20.0f + 2.0f * health, 31.0f, healthColor);
    AddCircle(238.0f, 25.5f, 7.0f, overrideActive ? Color{ 1.0f, 0.78f, 0.08f, 1.0f } : Color{ 0.22f, 0.55f, 0.88f, 1.0f });

    AddRect(kWidth - 226.0f, 15.0f, kWidth - 16.0f, 43.0f, { 0.03f, 0.08f, 0.14f, 0.95f });
    for (int i = 0; i < 12; ++i)
    {
        const bool filled = i < ((ammo + 1) / 2);
        AddRect(kWidth - 218.0f + i * 16.5f, 20.0f, kWidth - 207.0f + i * 16.5f, 36.0f,
            filled ? Color{ 1.0f, 0.72f, 0.18f, 1.0f } : Color{ 0.15f, 0.19f, 0.26f, 1.0f });
    }
    const float scoreFill = std::fmod(static_cast<float>(gPlayer->GetScore()), 1000.0f) / 1000.0f; // virtual
    AddRect(340.0f, 20.0f, 620.0f, 28.0f, { 0.07f, 0.12f, 0.20f, 0.95f });
    AddRect(340.0f, 20.0f, 340.0f + 280.0f * scoreFill, 28.0f, { 0.40f, 0.25f + speed * 0.08f, 0.98f, 1.0f });
}

void DrawActors()
{
    for (int i = 0; i < static_cast<int>(gEnemies.size()); ++i)
    {
        const Enemy& enemy = gEnemies[i];
        if (!enemy.alive || !gPlayer->IsEnemyVisible(i)) continue; // virtual wallhack target
        const Color glow = gPlayer->IsInvincible() ? Color{ 0.1f, 0.95f, 1.0f, 0.22f } : Color{ 1.0f, 0.12f, 0.18f, 0.20f };
        const Color body = enemy.hitFlash > 0.0f ? Color{ 1.0f, 0.9f, 0.3f, 1.0f } :
            (gPlayer->IsInvincible() ? Color{ 0.05f, 0.82f, 0.95f, 1.0f } : Color{ 0.90f, 0.16f, 0.24f, 1.0f });
        AddCircle(enemy.x, enemy.y, enemy.radius + 8.0f, glow, 20);
        AddCircle(enemy.x, enemy.y, enemy.radius, body, 16);
        AddCircle(enemy.x - enemy.radius * 0.28f, enemy.y - enemy.radius * 0.18f, 3.0f, { 1.0f, 1.0f, 1.0f, 0.9f }, 8);
    }

    for (const Bullet& bullet : gBullets)
    {
        if (bullet.alive) AddCircle(bullet.x, bullet.y, 4.0f, { 1.0f, 0.78f, 0.18f, 1.0f }, 8);
    }

    const float forwardX = gPlayer->aimX;
    const float forwardY = gPlayer->aimY;
    const float sideX = -forwardY;
    const float sideY = forwardX;
    const Color playerColor = gPlayer->IsInvincible() ? Color{ 1.0f, 0.78f, 0.08f, 1.0f } : Color{ 0.12f, 0.98f, 0.42f, 1.0f };
    if (gPlayer->IsInvincible()) AddCircle(gPlayer->x, gPlayer->y, 29.0f, { 1.0f, 0.78f, 0.08f, 0.20f }, 24);
    AddTriangle(gPlayer->x + forwardX * 24.0f, gPlayer->y + forwardY * 24.0f,
        gPlayer->x - forwardX * 13.0f + sideX * 16.0f, gPlayer->y - forwardY * 13.0f + sideY * 16.0f,
        gPlayer->x - forwardX * 13.0f - sideX * 16.0f, gPlayer->y - forwardY * 13.0f - sideY * 16.0f,
        playerColor);
    AddCircle(gPlayer->x, gPlayer->y, 7.0f, { 0.92f, 1.0f, 0.95f, 1.0f }, 10);
}
void DrawWastedOverlay()
{
    // Dark wash over the whole window.
    AddRect(0.0f, 0.0f, kWidth, kHeight, { 0.0f, 0.0f, 0.0f, 0.70f });

    // Red vignette: thin red border around the screen.
    const float border = 12.0f;
    AddRect(0.0f, 0.0f, kWidth, border, { 0.85f, 0.06f, 0.10f, 0.55f });
    AddRect(0.0f, kHeight - border, kWidth, kHeight, { 0.85f, 0.06f, 0.10f, 0.55f });
    AddRect(0.0f, 0.0f, border, kHeight, { 0.85f, 0.06f, 0.10f, 0.55f });
    AddRect(kWidth - border, 0.0f, kWidth, kHeight, { 0.85f, 0.06f, 0.10f, 0.55f });

    // Big game-over headline.
    // units; one character advance is 6 pixels (5 + 1 spacing).
    const float headlinePixel = 15.0f;
    const char headline[] = "GAME OVER";
    const float textWidth = static_cast<float>(sizeof(headline) - 1) * 6.0f * headlinePixel; // -1 to drop the '\0'
    const float textX = (kWidth - textWidth) * 0.5f;
    const float textY = (kHeight - 7.0f * headlinePixel) * 0.5f - 30.0f;
    DrawText(headline, textX, textY, headlinePixel, { 0.95f, 0.10f, 0.12f, 1.0f });

    // Subline: "SCORE 1234"
    char subline[32];
    std::snprintf(subline, sizeof(subline), "SCORE %d", gPlayer->GetScore());
    const float subPixel = 5.0f;
    const float subWidth = static_cast<float>(std::strlen(subline)) * 6.0f * subPixel;
    DrawText(subline, (kWidth - subWidth) * 0.5f, textY + 9.0f * headlinePixel + 12.0f, subPixel,
        { 0.85f, 0.86f, 0.92f, 1.0f });

    char bestLine[32];
    std::snprintf(bestLine, sizeof(bestLine), "BEST %d", gBestScore);
    const float bestWidth = static_cast<float>(std::strlen(bestLine)) * 6.0f * subPixel;
    DrawText(bestLine, (kWidth - bestWidth) * 0.5f, textY + 9.0f * headlinePixel + 52.0f, subPixel,
        { 1.0f, 0.72f, 0.18f, 1.0f });

    const char hint[] = "PRESS R TO RETRY";
    const float hintPixel = 4.0f;
    const float hintWidth = static_cast<float>(sizeof(hint) - 1) * 6.0f * hintPixel;
    DrawText(hint, (kWidth - hintWidth) * 0.5f, kHeight - 60.0f, hintPixel,
        { 0.70f, 0.74f, 0.80f, 0.85f });
}

void DrawPausedOverlay()
{
    AddRect(0.0f, 0.0f, kWidth, kHeight, { 0.0f, 0.0f, 0.0f, 0.58f });
    const char text[] = "PAUSED";
    const float pixel = 16.0f;
    const float width = static_cast<float>(sizeof(text) - 1) * 6.0f * pixel;
    DrawText(text, (kWidth - width) * 0.5f, (kHeight - 7.0f * pixel) * 0.5f, pixel,
        { 0.85f, 0.90f, 1.0f, 1.0f });
}
void Render()
{
    const float clearColor[4] = { 0.008f, 0.020f, 0.045f, 1.0f };
    gContext->OMSetRenderTargets(1, &gRenderTarget, nullptr);
    gContext->ClearRenderTargetView(gRenderTarget, clearColor);
    gVertices.clear();
    DrawWorld();
    DrawActors();
    DrawHud();
    if (gIsWasted) DrawWastedOverlay();
    else if (gIsPaused) DrawPausedOverlay();

    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (!gVertices.empty() && SUCCEEDED(gContext->Map(gVertexBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
    {
        std::memcpy(mapped.pData, gVertices.data(), gVertices.size() * sizeof(Vertex));
        gContext->Unmap(gVertexBuffer, 0);
        const UINT stride = sizeof(Vertex);
        const UINT offset = 0;
        const float blendFactor[4]{};
        gContext->IASetVertexBuffers(0, 1, &gVertexBuffer, &stride, &offset);
        gContext->IASetInputLayout(gInputLayout);
        gContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        gContext->VSSetShader(gVertexShader, nullptr, 0);
        gContext->PSSetShader(gPixelShader, nullptr, 0);
        gContext->RSSetState(gRasterizer);
        gContext->OMSetBlendState(gBlendState, blendFactor, 0xFFFFFFFF);
        gContext->Draw(static_cast<UINT>(gVertices.size()), 0);
    }
    gSwapChain->Present(1, 0);
}

void SpawnBullet()
{
    for (Bullet& bullet : gBullets)
    {
        if (bullet.alive) continue;
        bullet.x = gPlayer->x + gPlayer->aimX * 27.0f;
        bullet.y = gPlayer->y + gPlayer->aimY * 27.0f;
        bullet.velocityX = gPlayer->aimX * 560.0f;
        bullet.velocityY = gPlayer->aimY * 560.0f;
        bullet.life = 1.2f;
        bullet.alive = true;
        return;
    }
}

void InitialiseEnemies()
{
    gEnemies =
    {
        Enemy{ 100.0f, 120.0f,  68.0f,  42.0f, 16.0f, 0.0f, 0.0f, 100, true },
        Enemy{ 745.0f, 105.0f, -60.0f,  55.0f, 18.0f, 0.0f, 0.0f, 150, true },
        Enemy{ 870.0f, 255.0f, -72.0f, -34.0f, 15.0f, 0.0f, 0.0f, 100, true },
        Enemy{ 115.0f, 515.0f,  54.0f, -66.0f, 17.0f, 0.0f, 0.0f, 150, true },
        Enemy{ 545.0f, 285.0f,  75.0f,  31.0f, 16.0f, 0.0f, 0.0f, 200, true },
        Enemy{ 700.0f, 535.0f, -51.0f, -59.0f, 18.0f, 0.0f, 0.0f, 200, true },
        Enemy{ 290.0f, 250.0f,  63.0f, -48.0f, 15.0f, 0.0f, 0.0f, 250, true },
    };
}

void Reset()
{
    gIsWasted = false;
    gIsPaused = false;
    gPlayer->health = 100;
    gPlayer->score = 0;
    gPlayer->ammo = 24;
    gPlayer->x = 480.0f;
    gPlayer->y = 520.0f;
    gPlayer->aimX = 0.0f;
    gPlayer->aimY = -1.0f;
    gPlayer->hitFlash = 0.0f;
    gPlayer->fireCooldown = 0.0f;
    gPlayer->reloadTime = 0.0f;
    for (Bullet& bullet : gBullets) bullet.alive = false;
    InitialiseEnemies();
}

void UpdateAim(HWND window)
{
    POINT mouse{};
    GetCursorPos(&mouse);
    ScreenToClient(window, &mouse);
    const float dx = static_cast<float>(mouse.x) - gPlayer->x;
    const float dy = static_cast<float>(mouse.y) - gPlayer->y;
    const float length = std::sqrt(dx * dx + dy * dy);
    if (length > 2.0f)
    {
        gPlayer->aimX = dx / length;
        gPlayer->aimY = dy / length;
    }
}

void UpdateEnemies(float deltaSeconds)
{
    for (Enemy& enemy : gEnemies)
    {
        enemy.hitFlash = std::max(0.0f, enemy.hitFlash - deltaSeconds * 4.0f);
        if (!enemy.alive)
        {
            enemy.respawnTime -= deltaSeconds;
            if (enemy.respawnTime <= 0.0f)
            {
                enemy.alive = true;
                enemy.x = 80.0f + static_cast<float>((enemy.value * 37) % 780);
                enemy.y = 90.0f + static_cast<float>((enemy.value * 71) % 440);
            }
            continue;
        }

        const float speedMultiplier = 1.0f + std::min(gPlayer->GetScore(), 4000) * 0.00003f;
        enemy.x += enemy.velocityX * speedMultiplier * deltaSeconds;
        enemy.y += enemy.velocityY * speedMultiplier * deltaSeconds;
        if (enemy.x < enemy.radius + 18.0f || enemy.x > kWidth - enemy.radius - 18.0f) enemy.velocityX = -enemy.velocityX;
        if (enemy.y < enemy.radius + 62.0f || enemy.y > kHeight - enemy.radius - 18.0f) enemy.velocityY = -enemy.velocityY;
        for (const Rect& wall : kWalls)
        {
            if (PointInRect(enemy.x, enemy.y, wall))
            {
                enemy.velocityX = -enemy.velocityX;
                enemy.velocityY = -enemy.velocityY;
                enemy.x += enemy.velocityX * deltaSeconds * 2.0f;
                enemy.y += enemy.velocityY * deltaSeconds * 2.0f;
            }
        }

        const float dx = enemy.x - gPlayer->x;
        const float dy = enemy.y - gPlayer->y;
        const float touchingDistance = enemy.radius + 18.0f;
        if (dx * dx + dy * dy < touchingDistance * touchingDistance && !gPlayer->IsInvincible() && gPlayer->hitFlash <= 0.0f)
        {
            gPlayer->TakeDamage(8);
            enemy.velocityX = -enemy.velocityX;
            enemy.velocityY = -enemy.velocityY;
        }
    }
}

void UpdateBullets(float deltaSeconds)
{
    for (Bullet& bullet : gBullets)
    {
        if (!bullet.alive) continue;
        bullet.x += bullet.velocityX * deltaSeconds;
        bullet.y += bullet.velocityY * deltaSeconds;
        bullet.life -= deltaSeconds;
        if (bullet.life <= 0.0f) { bullet.alive = false; continue; }
        for (Enemy& enemy : gEnemies)
        {
            if (!enemy.alive) continue;
            const float dx = enemy.x - bullet.x;
            const float dy = enemy.y - bullet.y;
            if (dx * dx + dy * dy <= (enemy.radius + 4.0f) * (enemy.radius + 4.0f))
            {
                bullet.alive = false;
                enemy.alive = false;
                enemy.respawnTime = 2.1f;
                gPlayer->score += enemy.value;
                break;
            }
        }
    }
}

void UpdateWindowTitle(HWND window, float& titleTimer)
{
    titleTimer -= 0.016f;
    if (titleTimer > 0.0f) return;
    titleTimer = 0.25f;
    const std::wstring status = gPlayer->IsInvincible() ? L" | VTABLE OVERRIDE ACTIVE" : L" | WASD/Arrows move, Mouse+Space fire, P pause";
    const std::wstring wasted = gIsWasted ? L" | WASTED" : L"";
    const std::wstring paused = gIsPaused ? L" | PAUSED" : L"";
    const std::wstring title = std::wstring(L"SpaceProx — D3D11 Arena | HP ") + std::to_wstring(gPlayer->GetHealth()) +
        L" | Ammo " + std::to_wstring(gPlayer->GetAmmo()) + L" | Score " + std::to_wstring(gPlayer->GetScore()) +
        L" | Best " + std::to_wstring(gBestScore) + wasted + paused + status;
    SetWindowTextW(window, title.c_str());
}

LRESULT CALLBACK WindowProcedure(HWND window, UINT message, WPARAM wParam, LPARAM lParam)
{
    if (message == WM_DESTROY)
    {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(window, message, wParam, lParam);
}
} // namespace SpaceProx

using namespace SpaceProx;
int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int showCommand)
{
    WNDCLASSEXW windowClass{};
    windowClass.cbSize = sizeof(windowClass);
    windowClass.lpfnWndProc = WindowProcedure;
    windowClass.hInstance = instance;
    windowClass.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    windowClass.lpszClassName = kWindowClass;
    RegisterClassExW(&windowClass);

    RECT windowRect{ 0, 0, static_cast<LONG>(kWidth), static_cast<LONG>(kHeight) };
    HWND window = CreateWindowExW(0, kWindowClass, kWindowTitle, WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
        CW_USEDEFAULT, CW_USEDEFAULT, windowRect.right - windowRect.left, windowRect.bottom - windowRect.top,
        nullptr, nullptr, instance, nullptr);
    if (!window) return 1;

    InitialiseEnemies();
    if (!CreateD3D(window))
    {
        MessageBoxW(window, L"Could not initialise the D3D11 renderer.", L"SpaceProx", MB_ICONERROR);
        ReleaseD3D();
        return 1;
    }

    ShowWindow(window, showCommand);
    UpdateWindow(window);
    LARGE_INTEGER frequency{};
    LARGE_INTEGER previous{};
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&previous);
    float titleTimer = 0.0f;
    bool restartWasDown = false;
    bool pauseWasDown = false;
    MSG message{};
    while (message.message != WM_QUIT)
    {
        if (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&message);
            DispatchMessageW(&message);
            continue;
        }

        LARGE_INTEGER now{};
        QueryPerformanceCounter(&now);
        const float deltaSeconds = std::clamp(static_cast<float>(now.QuadPart - previous.QuadPart) / static_cast<float>(frequency.QuadPart), 0.0f, 0.05f);
        previous = now;

        if ((GetAsyncKeyState(VK_ESCAPE) & 0x8000) != 0)
        {
            PostMessageW(window, WM_CLOSE, 0, 0);
            continue;
        }

        const bool restartIsDown = ((GetAsyncKeyState('R') & 0x8000) != 0) || ((GetAsyncKeyState(VK_RETURN) & 0x8000) != 0);
        const bool pauseIsDown = (GetAsyncKeyState('P') & 0x8000) != 0;

        if (gIsWasted)
        {
            if (restartIsDown && !restartWasDown) Reset();
        }
        else
        {
            if (pauseIsDown && !pauseWasDown) gIsPaused = !gIsPaused;
            if (!gIsPaused)
            {
                UpdateAim(window);
                gPlayer->Update(deltaSeconds);
                if ((GetAsyncKeyState(VK_SPACE) & 0x8000) && gPlayer->fireCooldown <= 0.0f && gPlayer->ConsumeAmmo())
                {
                    SpawnBullet();
                    gPlayer->fireCooldown = 0.16f;
                }
                UpdateEnemies(deltaSeconds);
                UpdateBullets(deltaSeconds);
            }
        }
        restartWasDown = restartIsDown;
        pauseWasDown = pauseIsDown;
        UpdateWindowTitle(window, titleTimer);
        Render();
    }

    ReleaseD3D();
    return static_cast<int>(message.wParam);
}
