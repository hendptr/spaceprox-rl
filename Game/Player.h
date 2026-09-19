#pragma once

namespace SimpleGame7
{
// Keep this virtual-function order synchronized with GameHook7.cpp.
// No virtual destructor is intentional: Update is vtable[0].
class Player
{
public:
    Player();

    virtual void Update(float deltaSeconds);          // vtable[0]
    virtual void TakeDamage(int amount);              // vtable[1]
    virtual int GetHealth();                           // vtable[2]
    virtual bool IsInvincible();                       // vtable[3]
    virtual float GetSpeed();                          // vtable[4]
    virtual bool IsEnemyVisible(int enemyIndex);       // vtable[5]
    virtual int GetScore();                            // vtable[6]
    virtual int GetAmmo();                             // vtable[7]
    virtual bool ConsumeAmmo();                        // vtable[8]

    // NON-VIRTUAL method (NOT in the vtable). This is the target for the
    // INLINE / DETOUR hook in GameHook7. A non-virtual method has no vtable
    // slot, so it cannot be vtable-hooked.
    // noinline keeps it as a real, separate function so the inline hook on it
    // is actually reached (the compiler would otherwise inline it into
    // TakeDamage, and the hook would never fire).
    __declspec(noinline) int GetDamageTaken(int amount);

    // NOTE: No magic field. GameHook7 finds this object by vtable validation,
    // scanning the module for an object whose vtable lives entirely inside the
    // game executable.
    int health;
    int score;
    int ammo;
    float x, y;
    float aimX, aimY;
    float hitFlash;
    float fireCooldown;
    float reloadTime;
};

// Defined in a separate translation unit, keeping the concrete object opaque
// to the renderer and preserving virtual dispatch in optimized builds.
__declspec(noinline) Player* CreatePlayer();
} // namespace SimpleGame7