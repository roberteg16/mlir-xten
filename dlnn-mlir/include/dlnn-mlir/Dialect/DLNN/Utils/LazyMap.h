/// Declares the LazyMap utility type.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@amd.com)

#pragma once

#include "llvm/ADT/DenseMap.h"

#include <optional>

namespace mlir::dlnn {

/// Base class for an immutable map that lazily populates itself.
template<class Key, class Value>
class LazyMap : llvm::DenseMap<Key, std::optional<Value>> {
    using Base = llvm::DenseMap<Key, std::optional<Value>>;

public:
    using typename Base::value_type;
    using typename Base::size_type;
    using typename Base::key_type;
    using typename Base::mapped_type;
    using iterator = typename Base::const_iterator;

    virtual ~LazyMap() {}

    using Base::empty;
    using Base::size;
    iterator begin() const { return Base::begin(); }
    iterator end() const { return Base::end(); }

    /// Looks up a cached entry for @p key .
    iterator find(const Key &key) const { return Base::find(key); }
    /// @copydoc find(const Key &)
    const std::optional<Value> &lookup(const Key &key)
    {
        const auto it = find(key);
        if (it != end()) return it->getSecond();
        return m_tombstone;
    }
    /// Gets or computes the entry for @p key .
    const std::optional<Value> &getOrCompute(const Key &key)
    {
        if (const auto &maybe = lookup(key)) return maybe;
        if (auto maybe = compute(key)) {
            auto [it, ok] =
                Base::try_emplace(key, std::in_place, std::move(*maybe));
            assert(ok);
            return it->getSecond();
        }
        return m_tombstone;
    }

    /// Removes any cached mapping for @p key .
    bool evict(const Key &key) { return erase(key); }

protected:
    /// Replaces any stored mapping for @p key by emplacing a new value.
    template<class... Args>
    Value &replace(const Key &key, Args &&... args)
    {
        return replace_impl(key, std::in_place, std::forward<Args>(args)...);
    }
    /// Caches the empty option for @p key .
    void erase(const Key &key) { replace_impl(key); }

    /// When overridden in a derived class, computes the entry for @p key .
    virtual std::optional<Value> compute(const Key &key) const = 0;

private:
    const std::optional<Value> m_tombstone;

    template<class... Args>
    Value &replace_impl(const Key &key, Args &&... args)
    {
        auto [it, ok] = Base::try_emplace(key, std::forward<Args>(args)...);
        if (ok) return &it->getSecond();

        it->getSecond().~mapped_type();
        return *::new ((void*)&it->getSecond())
            mapped_type(std::forward<Args>(args)...);
    }
};

} // namespace mlir::dlnn
